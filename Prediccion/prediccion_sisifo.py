# Importamos la biblioteca TensorFlow para trabajar con redes neuronales y aprendizaje profundo.
import tensorflow as tf

# Importamos varias bibliotecas estándar y de terceros para diversas funcionalidades.

# Bibliotecas para manejo de archivos y tiempo.
import os
import time
import datetime
import random
import glob

# Bibliotecas numéricas y de análisis de datos.
import numpy as np
import pandas as pd
import scipy as sp

# Biblioteca para interactuar con bases de datos SQL.
from sqlalchemy import create_engine
import psycopg2

# Biblioteca para cargar variables de entorno desde archivos .env.
from dotenv import load_dotenv

load_dotenv()

# Conectar a servidor de CCT en Psql
psql_user = os.getenv('user_azure')
psql_password = os.getenv('password_azure')
psql_localhost = os.getenv('localhost_azure')
psql_port = os.getenv('port_azure')
psql_database = os.getenv('database_azure')
psql_engine = create_engine(f"postgresql://{psql_user}:{psql_password}@{psql_localhost}:{psql_port}/{psql_database}?sslmode=require")
psql_connection = psql_engine.connect()


def delete_table_month(table_name,date_min,date_max):
    q_ag_fc_dt_dia = f'''          
    DELETE FROM {table_name} WHERE  (               
        fecha_hora >= '{date_min}' and
        fecha_hora <= '{date_max}'
    )
    '''
    conn = psycopg2.connect(database=psql_database,host=psql_localhost,user=psql_user, password=psql_password, port=psql_port)
    cursor = conn.cursor()
    cursor.execute(q_ag_fc_dt_dia)
    conn.commit()
    conn.close()

def df_psql_batch(df,table_name):    
    try:
        batch_size = 5000
        for start in range(0,len(df),batch_size):
            end = start + batch_size 
            df_subset = df.iloc[start:end]
            df_subset.to_sql(table_name, psql_engine, if_exists='append', index=False, method="multi",chunksize=5000) 
        return True
    except Exception as ex:
        print("error",ex)
        return False



def calculate_df(rail, date_min, date_max,date_max_pred):

    filenamecsv = 'fc_dt/fc_dt_' + str(rail) +'_' + date_min.strftime("%Y_%m") + '_'+ date_max.strftime("%Y_%m")  + '.csv'
    if os.path.isfile(filenamecsv) == True:
        df = pd.read_csv(filenamecsv, delimiter=";",na_values=' ',decimal=',', converters={'carril': str})
    else:
        template_sql = f"""
        SELECT fecha_hora AS date_hour, 
        intensidad AS value_intensity_real, 
        velocidad AS value_speed_real
        FROM fc_dt
        WHERE fecha_hora >= '{date_min}' 
        AND fecha_hora <= '{date_max}'
        and carril = '{rail}'
        order by fecha_hora asc
        """
        df=pd.read_sql(sql=template_sql,con=psql_connection)
        df.to_csv(filenamecsv,index=False,sep=';',decimal=',')

    # Manipulación y limpieza de los datos en el DataFrame
    df['date_hour'] = pd.to_datetime(df["date_hour"])
    df['date_hour'] = df['date_hour'].dt.round('1h')
    df.drop_duplicates(subset=['date_hour'], inplace=True)
    
    # Crear un DataFrame temporal con un rango de fechas
    df_tmp = pd.DataFrame(dict(date_hour=pd.date_range(start=date_min, end=date_max_pred, freq='H')))
    df_tmp['date_hour'] = pd.to_datetime(df_tmp['date_hour'])
    df_tmp['rail'] = int(rail)

    # Agregar columnas de tiempo para análisis temporal posterior
    df_tmp['year'] = [i.year for i in df_tmp['date_hour']]
    df_tmp['month'] = [i.month for i in df_tmp['date_hour']]
    df_tmp['day'] = [i.day for i in df_tmp['date_hour']]
    df_tmp['hour'] = [i.hour for i in df_tmp['date_hour']]

    df_tmp['day_of_week'] = [i.dayofweek for i in df_tmp['date_hour']]
    df_tmp['week_of_year'] = [i.week for i in df_tmp['date_hour']]
    df_tmp['day_of_year'] = [i.dayofyear for i in df_tmp['date_hour']]

    # Combinar DataFrames por fecha
    df = pd.merge(df_tmp, df, on='date_hour', how='left')
    
    return df
    
   
def calculate_data(df):
    """
    Realiza el cálculo y preprocesamiento de los datos en el DataFrame.

    Este método toma el DataFrame almacenado en el objeto y realiza las siguientes operaciones de preprocesamiento:
    1. Elimina filas con valores nulos.
    2. Elimina filas con valores atípicos basados en la puntuación Z.
    3. Almacena el DataFrame preprocesado en el atributo 'df_data' del objeto.

    Args:
        *args: Argumentos posicionales opcionales.
        **kwargs: Argumentos clave opcionales.

    Returns:
        None
    """
    
    df=df.copy()
    
    # Eliminar los datos null
    df.dropna(inplace=True)
    
    # Eliminar los datos atipicos
    df = df[ (np.abs(sp.stats.zscore(df['value_intensity_real'])) <= 3.0) & (np.abs(sp.stats.zscore(df['value_speed_real'])) <= 3.0)]
    
    # Almacenar el DataFrame con información limpia de nulos y datos atipicos para el entrenamiento del modelo
    return df

def calculate_mdl(df_data,target_columns,feature_columns,rail):
    """
    Calcula el Modelo de Aprendizaje Profundo (MDL) utilizando una red neuronal secuencial.

    Este método construye una red neuronal secuencial con capas densas y lo entrena con los datos
    de características y objetivos proporcionados en el DataFrame df_data. Utiliza la función de pérdida
    de error absoluto medio (MAE) y el optimizador Adam para el entrenamiento.

    Args:
        args: Argumentos posicionales adicionales (no se utilizan en este método).
        kwargs: Argumentos clave adicionales (no se utilizan en este método).

    Returns:
        None

    Nota:
        Este método modifica los siguientes atributos de la instancia:
        - score: El valor de la métrica de error absoluto medio (MAE) del modelo en los datos de entrenamiento.
        - model: El modelo de red neuronal entrenado.

    """

    model_file = 'models/' + str(rail) + '.keras'
    
    if os.path.isfile(model_file) == True:
        model = tf.keras.models.load_model(model_file)
        score = model.evaluate(df_data[feature_columns], df_data[target_columns])

    else:
        n_outputs=len(target_columns)
        
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(n_outputs, activation='linear')
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.metrics.mae, metrics=[tf.keras.metrics.mae])

        model.fit(df_data[feature_columns], df_data[target_columns], epochs=100, batch_size=25, validation_split=0.2, validation_steps=10, verbose=0)

        score = model.evaluate(df_data[feature_columns], df_data[target_columns])


        model.save(model_file)  # The file needs to end with the .keras extension       

    return model,score
    
def calculate_predict(df,model, feature_columns,predict_columns,rail):
    df_final=pd.concat([df.reset_index(drop=True), pd.DataFrame(data=model.predict(df[feature_columns]), columns=predict_columns)],axis=1)

    filenamecsv = 'predictions/fc_dt_pred_'+ str(rail) +'_' + date_min.strftime("%Y_%m") + ''+ date_max.strftime("%Y_%m")  + '.csv'
    #if os.path.isfile(filenamecsv) == False:
    df_final.to_csv(filenamecsv,index=False,sep=';',decimal=',')

    return df_final
    
def calculate_info(df_final,model,rail,score):
    """
    Realiza cálculos estadísticos y agrega información a un DataFrame.
    
    Este método calcula diversas estadísticas y agregaciones sobre los datos en un DataFrame
    y almacena los resultados en otro DataFrame llamado 'df_info'. Las estadísticas calculadas
    incluyen valores nulos, valores atípicos y agregaciones de datos en función de distintas
    métricas y atributos.
    
    Args:
        *args: Argumentos posicionales (no se utilizan en este método).
        **kwargs: Argumentos clave (no se utilizan en este método).
        
    Returns:
        None
    
    """
    groups = [
        "rail",
    ]

    # Funciones personalizadas para cálculos estadísticos
    
    # Funcion para calcular los datos nulos
    def sum_isnan(x):
        return np.isnan(x).sum()
    
    # Funcion para calcular los porcentaje de datos nulos
    def percent_isnan(x):
        return np.isnan(x).sum() * 100 / np.size(x)
    
    # Funcion para calcular los la cantidad de datos atipicos
    def noutlier(x):
        return np.array(np.where(np.abs(sp.stats.zscore(x[~np.isnan(x)])) > 3)).size
    
    # Funcion para calcular los la cantidad porcentual de datos atipicos
    def percent_noutlier(x):
        return np.array(np.where(np.abs(sp.stats.zscore(x[~np.isnan(x)])) > 3)).size / np.size(x)

    # Definición de agregaciones para el análisis estadístico
    aggs = dict(
        date_hour=['min', 'max'],
        value_intensity_real=['sum', 'min', 'mean', 'max', 'count', 'size', sum_isnan, percent_isnan, noutlier, percent_noutlier],
        value_speed_real=['sum', 'min', 'mean', 'max', 'count', 'size', sum_isnan, percent_isnan, noutlier, percent_noutlier],
        value_intensity_predict=['sum', 'min', 'mean', 'max', 'count', 'size', sum_isnan, percent_isnan, noutlier, percent_noutlier],
        value_speed_predict=['sum', 'min', 'mean', 'max', 'count', 'size', sum_isnan, percent_isnan, noutlier, percent_noutlier],
    )

    # Copiar DataFrame y realizar cálculos estadísticos
    df = df_final.copy()
    df = df.groupby(groups).agg(aggs).reset_index()
    
    df.columns = df.columns.map("_".join).str.strip("_")
    df['mae']=score[0]
    
    # Almacenar el DataFrame con información estadística
    filenamecsv = 'models/stat/'+ str(rail) +'_' + date_min.strftime("%Y_%m") + ''+ date_max.strftime("%Y_%m")  + '.csv'
    #if os.path.isfile(filenamecsv) == False:
    df.to_csv(filenamecsv,index=False,sep=';',decimal=',')

    df_psql_batch(df,"dm_fc_dt_models")



def prediction_fc_dt(date_min,date_max,date_max_pred):

    template_sql = f"""
    SELECT distinct carril  as carril 
    FROM dm_carril
    where carril not in (select distinct rail as carril from fc_dt_pred)    
    """
    df=pd.read_sql(sql=template_sql,con=psql_connection)

    carril_ids=df["carril"].unique().tolist()
    print("carril_ids",carril_ids)
    for rail in carril_ids:
        print("rail",rail)
        dfr=df.query("carril==@rail") 

        # Asignar un código de carril a la variable 'rail'
        rail = rail#'0303134'

        # Definir la fecha mínima y máxima para el análisis
        #date_min = datetime.datetime(2022, 6, 1, 0)  # 1 de junio de 2022 a las 00:00
        #date_max = datetime.datetime(2023, 6, 1, 0)  # 1 de junio de 2023 a las 00:00

        # Definir las columnas de características que se utilizarán en el modelo
        feature_columns = [
            'week_of_year',    # Semana del año
            'day_of_week',     # Día de la semana (0: lunes, 6: domingo)
            'hour',            # Hora del día (0-23)
        ]

        # Definir las columnas objetivo (valores reales) que se utilizarán en el modelo
        target_columns = [
            'value_intensity_real',    # Valor real de la intensidad
            'value_speed_real',        # Valor real de la velocidad
        ]

        # Definir las columnas de predicción que se generarán a través del modelo
        predict_columns = [
            'value_intensity_predict',    # Valor predicho de la intensidad
            'value_speed_predict',        # Valor predicho de la velocidad
        ]

        df_cdf = calculate_df(rail,date_min,date_max,date_max_pred)
        if df_cdf['value_intensity_real'].max() > 0:
            df_cd = calculate_data(df_cdf)
            model,score = calculate_mdl(df_cd,target_columns,feature_columns,rail)
            df_final = calculate_predict(df_cdf,model, feature_columns,predict_columns,rail)
            print("score",score)         
            #df_psql_batch(df_final,"fc_dt_pred")
            calculate_info(df_final,model,rail,score)


date_min = datetime.datetime(2022, 1, 1, 0)  # 1 de junio de 2022 a las 00:00
date_max = datetime.datetime(2023, 12, 15, 0) 
date_max_pred = datetime.datetime(2025, 1, 1, 0) 

prediction_fc_dt(date_min,date_max,date_max_pred)
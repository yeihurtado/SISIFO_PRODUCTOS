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

"""
Elimina filas de una tabla en una base de datos PostgreSQL en función de un rango de fechas.

Esta función se conecta a una base de datos PostgreSQL utilizando las credenciales almacenadas en variables de entorno 
y elimina filas de la tabla especificada donde la columna 'fecha_hora' (fecha-hora) se encuentra dentro del 
rango de fechas proporcionado (inclusivo).

Parámetros:
    table_name (str): El nombre de la tabla de la que se eliminarán las filas.
    date_min (datetime.datetime): La fecha mínima (inclusiva) para el rango de eliminación.
    date_max (datetime.datetime): La fecha máxima (inclusiva) para el rango de eliminación.

Devuelve:
    Ninguno
"""
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

"""
Inserta un DataFrame de Pandas en una tabla PostgreSQL en lotes.

Esta función toma un DataFrame de Pandas (`df`) y el nombre de una tabla PostgreSQL (`table_name`) como entrada.
Divide el DataFrame en lotes de tamaño especificado y los inserta en la tabla de forma eficiente utilizando la conexión
a la base de datos establecida mediante SQLAlchemy.

Argumentos:
    df (pandas.DataFrame): El DataFrame de Pandas que contiene los datos a insertar.
    table_name (str): El nombre de la tabla PostgreSQL en la que se insertarán los datos.

Devuelve:
    bool: True si la inserción se realizó correctamente, False si se produjo un error.
"""
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
def calculate_data(df):

    
    df=df.copy()
    
    # Eliminar los datos null
    df.dropna(inplace=True)
    
    # Eliminar los datos atipicos
    df = df[ (np.abs(sp.stats.zscore(df['value_intensity_real'])) <= 3.0) & (np.abs(sp.stats.zscore(df['value_speed_real'])) <= 3.0)]
    
    # Almacenar el DataFrame con información limpia de nulos y datos atipicos para el entrenamiento del modelo
    return df


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
def calculate_mdl(df_data,target_columns,feature_columns,rail):


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
    

"""
Realiza predicciones para el DataFrame utilizando un modelo de aprendizaje automático.

Esta función toma un DataFrame preprocesado (`df`), un modelo de aprendizaje automático entrenado (`model`), 
listas de nombres de columnas para características (`feature_columns`) y predicciones (`predict_columns`), 
y el identificador del carril (`rail`) como entrada. La función realiza lo siguiente:

1. **Preparación del DataFrame para la predicción:**
    - Combina el DataFrame original (`df`) con las predicciones realizadas por el modelo.
    - El modelo se aplica a las columnas especificadas en `feature_columns` del DataFrame y las predicciones se guardan 
    en un nuevo DataFrame con las columnas especificadas en `predict_columns`.
    - La función `pd.concat` se utiliza para concatenar el DataFrame original con el DataFrame de predicciones a lo largo 
    del eje 1 (columnas).

2. **Generación del nombre del archivo CSV (comentado):**
    - Construye el nombre del archivo CSV basado en el carril (`rail`), fechas mínima y máxima (asumiendo su existencia 
    en variables `date_min` y `date_max`) y prefijos para diferenciarlo de los archivos sin predicciones.
    - Esta parte está comentada, lo que indica que la función actualmente no guarda el DataFrame con las predicciones.

3. **Retorno del DataFrame:**
    - La función devuelve el DataFrame final (`df_final`) que contiene los datos originales combinados con las columnas de predicción.



**Argumentos:**

* `df` (pandas.DataFrame): El DataFrame preprocesado que contiene los datos para realizar las predicciones.
* `model` (Machine Learning Model): El modelo de aprendizaje automático entrenado que se utilizará para realizar las predicciones.
* `feature_columns` (list): Una lista de nombres de columna que representan las características que se usarán para la predicción.
* `predict_columns` (list): Una lista de nombres de columna para las predicciones realizadas por el modelo.
* `rail` (str): Identificador del carril del que se procesaron los datos.

**Devuelve:**

* `pandas.DataFrame`: El DataFrame final que contiene los datos originales junto con las columnas de predicción generadas por el modelo.
"""
def calculate_predict(df,model, feature_columns,predict_columns,rail):
    df_final=pd.concat([df.reset_index(drop=True), pd.DataFrame(data=model.predict(df[feature_columns]), columns=predict_columns)],axis=1)

    filenamecsv = 'predictions/fc_dt_pred_'+ str(rail) +'_' + date_min.strftime("%Y_%m") + ''+ date_max.strftime("%Y_%m")  + '.csv'
    #if os.path.isfile(filenamecsv) == False:
    df_final.to_csv(filenamecsv,index=False,sep=';',decimal=',')

    return df_final
    

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
def calculate_info(df_final,model,rail,score):
    
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


"""
Realiza predicciones de intensidad y velocidad vehicular para carriles específicos en un rango de fechas.

Esta función toma como entrada la fecha mínima (`date_min`), la fecha máxima (`date_max`) y la fecha máxima del rango de predicción (`date_max_pred`). 
La función realiza los siguientes pasos:

1. **Identificación de carriles sin predicciones:**
    - Ejecuta una consulta SQL para identificar los carriles que no tienen predicciones existentes en la tabla `fc_dt_pred`.
    - Almacena los identificadores de los carriles sin predicciones en una lista (`carril_ids`).

2. **Predicción para cada carril:**
    - Recorre la lista de `carril_ids` para procesar cada carril individualmente.
    - Para cada carril (`rail`):
        - Filtra el DataFrame original (`df`) para obtener los datos específicos del carril actual.
        - Define las columnas de características (`feature_columns`) y las columnas objetivo (`target_columns`) para el modelo.
        - Define las columnas de predicción (`predict_columns`) que se generarán a través del modelo.
        - Calcula el DataFrame con los datos del carril y las fechas del rango de predicción (`df_cdf`).
        - Verifica si existen valores reales de intensidad en el DataFrame (`df_cdf['value_intensity_real'].max() > 0`).
            - Si existen valores reales de intensidad:
                - Calcula el conjunto de datos preprocesado (`df_cd`) para el modelo.
                - Entrena un modelo de aprendizaje automático (`model`) y obtiene su puntuación (`score`).
                - Realiza predicciones con el modelo entrenado y genera el DataFrame final (`df_final`) con las predicciones.
                - Imprime la puntuación del modelo (`score`).
                - **(Comentado)** Guarda el DataFrame final con las predicciones en la tabla `fc_dt_pred` (línea comentada).
                - Calcula y guarda información adicional sobre el modelo y las predicciones para el carril (`rail`).

3. **Retorno:**
    - La función no devuelve explícitamente ningún valor.
    - Guarda el DataFrame final con las predicciones en la tabla `fc_dt_pred` (línea comentada).
    - Genera información adicional sobre el modelo y las predicciones para cada carril procesado.

**Argumentos:**

* `date_min` (datetime.datetime): Fecha mínima del rango de datos para el análisis.
* `date_max` (datetime.datetime): Fecha máxima del rango de datos para el análisis.
* `date_max_pred` (datetime.datetime): Fecha máxima del rango de predicción.

**Devoluciones:**

* No hay devoluciones explícitas.
* Guarda el DataFrame final con las predicciones en la tabla `fc_dt_pred` (línea comentada).
* Genera información adicional sobre el modelo y las predicciones para cada carril procesado.

**Consideraciones:**

* La función asume la existencia de tablas `dm_carril` y `fc_dt_pred` en la base de datos PostgreSQL.
* La función utiliza funciones auxiliares `calculate_df`, `calculate_data`, `calculate_mdl`, `calculate_predict` y `calculate_info` 
que se presumen definidas en otro lugar del código.
* La línea que guarda el DataFrame final en la tabla `fc_dt_pred` está actualmente comentada.
* La función podría mejorarse para manejar casos donde no existen valores reales de intensidad para un carril.

"""
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
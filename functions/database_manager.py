import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argon2
import pyodbc
import xlsxwriter
from dotenv import load_dotenv
from utils.settings_controller import CREATE_PATH
from colorama import Fore, Style, init
from datetime import date, datetime, time, timedelta
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN
from functions.encrypted import *
import struct
# Cargar variables de entorno desde .env
init()
load_dotenv()

COLD_DOWN_ASISTENCIA = 5
def get_config_value(key, default=None):
    return os.getenv(key, default)

APP_ENV = get_config_value("APP_ENV", "production")


#-------------------------------------------------------------------------------------------------------------------------------------
#CONEXIONES
#-------------------------------------------------------------------------------------------------------------------------------------
class DatabaseManager:
    def __init__(self):
        try:
            if APP_ENV == "testing":
                # pruebas
                self.connection = pyodbc.connect(
                    "DRIVER={ODBC Driver 17 for SQL Server};"
                    "SERVER=DESKTOP-8I672CP;"
                    "DATABASE=dbRealSecurity;"
                    "Trusted_Connection=yes;"
                    "TrustServerCertificate=yes;"
                )
            else:
                # produccion
                self.connection = pyodbc.connect(
                    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                    f"SERVER={get_config_value('DB_HOST')},{get_config_value('DB_PORT')};"
                    f"DATABASE={get_config_value('DB_NAME')};"
                    f"UID={get_config_value('DB_USER')};"
                    f"PWD={get_config_value('DB_PASSWORD')};"
                    f"TrustServerCertificate=yes;"
                )
            # print(f"{Fore.GREEN}Conexión exitosa usando: {APP_ENV} {Style.RESET_ALL}")
            self.create_database_if_not_exists()
        except pyodbc.Error as e:
            raise RuntimeError(f"{Fore.RED} Error al conectar con SQL Server: {e} - entorno {APP_ENV} {Style.RESET_ALL}")

    def create_database_if_not_exists(self):
        try:
            cursor = self.connection.cursor()

            cursor.execute("IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'dbRealSecurity') EXEC('CREATE DATABASE dbRealSecurity');")
            self.connection.commit()

            script_path = os.path.join(os.path.dirname(__file__), CREATE_PATH)

            if not os.path.exists(script_path):
                print(f"{Fore.YELLOW} Archivo de configuracion no encontrado: {script_path} {Style.RESET_ALL}")
                return

            with open(script_path, "r", encoding="utf-8") as file:
                sql_script = file.read()
            sql_statements = sql_script.split(";")
            for statement in sql_statements:
                statement = statement.strip()
                if statement:
                    try:
                        cursor.execute(statement)
                    except pyodbc.Error as e:
                        print(f"{Fore.RED} Error ejecutando SQL: {statement[:100]} {e} {Style.RESET_ALL}")

            self.connection.commit()
            # print(f"{Fore.GREEN}Base de datos y tablas creadas{Style.RESET_ALL}")
        except pyodbc.Error as e:
            print(f"{Fore.RED} Error al crear la base de datos: {e} {Style.RESET_ALL}")

#-------------------------------------------------------------------------------------------------------------------------------------
#EMPLEADOS
#-------------------------------------------------------------------------------------------------------------------------------------
    def create_empleado(self, cedula, apellidos, nombres, mail, phone, fechaN, genero, area):
        try:
            cursor = self.connection.cursor()
            cursor.execute(
            """
                INSERT INTO empleados (ci_emp, apellidos_emp, nombres_emp, mail_emp, phone_emp, fechaN_emp, genero_emp, area_emp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
                           , (cedula, apellidos, nombres, mail, phone, fechaN, genero, area))
            
            self.connection.commit()
            print(f"{Fore.GREEN} Empleado {nombres} {apellidos} agregado correctamente {Style.RESET_ALL}")
        except pyodbc.Error as e:
            print(f"{Fore.RED} Error al agregar empleado: {e} {Style.RESET_ALL}")


    def get_empleados(self):
        try:
            cursor = self.connection.cursor()
            cursor.execute(" SELECT * FROM empleados order by estado_emp desc;")
            empleados = cursor.fetchall()
            return empleados
        except pyodbc.Error as e:
            print(f"{Fore.RED} Error al obtener empleados: {e} {Style.RESET_ALL}")
            return []
        

    def get_empleado_ci(self, ci):
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM empleados WHERE ci_emp = ?", (ci,))
            empleado = cursor.fetchone()
            return empleado
        except pyodbc.Error as e:
            print(f"{Fore.RED} Error al obtener empleado con CI {ci}: {e} {Style.RESET_ALL}")
            return None
        
    def get_end_empleado(self):
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT TOP 1 id FROM empleados ORDER BY id DESC;")
            empleado = cursor.fetchone()
            return empleado[0] if empleado else None  # Devuelve solo el ID o None si no hay empleados
        except pyodbc.Error as e:
            print(f"{Fore.RED} Error al obtener último empleado: {e} {Style.RESET_ALL}")
            return None

        
        
    def update_empleado(self, emp_id, apellidos, nombres, correo, telefono, fecha_nacimiento, genero, area):
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                UPDATE empleados
                SET apellidos_emp = ?, nombres_emp = ?, mail_emp = ?, phone_emp = ?, fechaN_emp = ?, genero_emp = ?, area_emp = ?
                WHERE id = ?
            """, (apellidos, nombres, correo, telefono, fecha_nacimiento, genero, area, emp_id))
            
            self.connection.commit()
            print(f"{Fore.GREEN} Empleado ID {emp_id} actualizado correctamente {Style.RESET_ALL}")
        except pyodbc.Error as e:
            print(f"{Fore.RED} Error al actualizar empleado ID {emp_id}: {e} {Style.RESET_ALL}")


    def delete_empleado(self, ci):    
        try:
            cursor = self.connection.cursor()
            cursor.execute("DELETE FROM empleados WHERE ci_emp = ?", (ci,))
            self.connection.commit()
            print(f"{Fore.GREEN} Empleado con CI {ci} eliminado correctamente {Style.RESET_ALL}")
        except pyodbc.Error as e:
            print(f"{Fore.RED} Error al eliminar empleado con CI {ci}: {e} {Style.RESET_ALL}")




#-------------------------------------------------------------------------------------------------------------------------------------
#FOTO EMPLEADOS
#-------------------------------------------------------------------------------------------------------------------------------------
    def add_foto_emp(self, emp_id, foto_path):
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                INSERT INTO empleado_fotos (emp_id, ruta_carpeta) VALUES (?, ?)
                """
            , (emp_id, foto_path))
            
            self.connection.commit()
            print(f"{Fore.GREEN} Foto guardada para empleado ID {emp_id} {Style.RESET_ALL}")
        except pyodbc.Error as e:
            print(f"{Fore.RED} Error al guardar la foto del empleado ID {emp_id}: {e} {Style.RESET_ALL}")


    def get_foto_emp_id_emp(self, emp_id):
        try:    
            cursor = self.connection.cursor()
            cursor.execute(
                """
                SELECT * FROM empleado_fotos WHERE emp_id = ?
                """
            , (emp_id,))
            fotos = cursor.fetchall()
            return fotos
        except pyodbc.Error as e:
            print(f"{Fore.RED} Error al obtener fotos del empleado ID {emp_id}: {e} {Style.RESET_ALL}")
            return []
        

    def delete_foto_emp(self, foto_id):
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                DELETE FROM empleado_fotos WHERE id = ?
                """
            , (foto_id,))
            self.connection.commit()
            print(f"{Fore.GREEN} Foto ID {foto_id} eliminada correctamente {Style.RESET_ALL}")
        except pyodbc.Error as e:
            print(f"{Fore.RED} Error al eliminar la foto ID {foto_id}: {e} {Style.RESET_ALL}")




#-------------------------------------------------------------------------------------------------------------------------------------
#EMBEDDINGS
#-------------------------------------------------------------------------------------------------------------------------------------
    def add_embedding(self, emp_id, embedding):
        """
        Guarda un embedding en la base de datos en formato VARBINARY(MAX).
        """
        try:
            if not isinstance(embedding, np.ndarray) or embedding.shape != (512,):
                print(f" Error: El embedding proporcionado no es válido. Forma: {embedding.shape}")
                return

            embedding = embedding.astype(np.float32)

            # 🔹 Verificar normalización
            norm = np.linalg.norm(embedding)
            if norm == 0:
                print(f"⚠️ Error: El embedding tiene norma 0, lo que indica un problema.")
                return
            if not np.isclose(norm, 1.0, atol=1e-5):
                print(f"⚠️ Advertencia: Normalizando embedding, ya que su norma es {norm}")
                embedding = embedding / norm  # Normalizar si no está correctamente ajustado

            embedding_bin = embedding.tobytes()

            with self.connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO embeddings (emp_id, embedding) VALUES (?, ?)", 
                    (emp_id, embedding_bin)
                )
                self.connection.commit()
                print(f"✅ Embedding guardado correctamente para empleado ID {emp_id}.")

        except pyodbc.Error as e:
            print(f" Error SQL al guardar embedding del empleado ID {emp_id}: {e}")









    def get_embeddings(self):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT e.nombres_emp, e.apellidos_emp, emb.embedding
                    FROM empleados e
                    INNER JOIN embeddings emb ON e.id = emb.emp_id
                    WHERE e.estado_emp = 1 AND emb.estado_emb = 1
                """)
                resultados = cursor.fetchall()

            embeddings = []
            nombres = []

            for nombre, apellido, embedding_bytes in resultados:
                try:
                    if isinstance(embedding_bytes, bytes):
                        embedding_array = np.frombuffer(embedding_bytes, dtype=np.float32)

                        # 📌 Verificar que el embedding tenga 512 dimensiones
                        if embedding_array.shape == (512,):
                            # 🔹 Verificar la norma del embedding
                            norm = np.linalg.norm(embedding_array)
                            if norm == 0:
                                print(f"⚠️ Embedding de {nombre} {apellido} tiene norma 0. Será ignorado.")
                                continue
                            if not np.isclose(norm, 1.0, atol=1e-5):
                                print(f"⚠️ Normalizando embedding de {nombre} {apellido}. Norma antes: {norm}")
                                embedding_array = embedding_array / norm  # Normalización
                            
                            embeddings.append(embedding_array)
                            nombres.append(f"{nombre} {apellido}")
                        else:
                            print(f"⚠️ Embedding inválido con forma {embedding_array.shape}, se omite.")
                    else:
                        print(f" Error: el formato del embedding no es bytes. Se omitirá.")

                except Exception as e:
                    print(f" Error al procesar un embedding de {nombre} {apellido}: {e}")

            print(f"✅ {len(embeddings)} embeddings cargados correctamente y normalizados.")
            return embeddings, nombres

        except pyodbc.Error as e:
            print(f" Error al obtener embeddings desde la base de datos: {e}")
            return [], []




        

    def delete_foto_emp_ci_emp(self, emp_id):
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                DELETE FROM empleado_fotos WHERE emp_id = ?
                """
            , (emp_id,))
            self.connection.commit()
            print(f"{Fore.GREEN} Todas las fotos del empleado ID {emp_id} han sido eliminadas {Style.RESET_ALL}")
        except pyodbc.Error as e:
            print(f"{Fore.RED} Error al eliminar fotos del empleado ID {emp_id}: {e} {Style.RESET_ALL}")




#
#MODELO arcface
#

    # 📌 Guardar embedding en la base de datos (convertido a binario)
    def store_embedding(self, emp_id, embedding):
        embedding = self.normalize_embedding(embedding)  # Asegurar normalización
        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()

        with self.connection.cursor() as cursor:
            cursor.execute("INSERT INTO embeddings (emp_id, embedding) VALUES (?, ?)", (emp_id, embedding_bytes))
            self.connection.commit()







    def get_embeddings_arc(self):
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT embedding, emp_id FROM embeddings")  # ✅ Cambiar el orden para obtener primero el embedding
            results = cursor.fetchall()

        embeddings = []
        nombres_empleados = []

        for embedding_bytes, emp_id in results:  # ✅ Orden corregido para asegurar que embedding_bytes recibe el binario
            try:
                # 📌 Si el dato ya es un `numpy.ndarray`, no intentar convertirlo de nuevo
                if isinstance(embedding_bytes, np.ndarray):
                    print(f"⚠️ Embedding ya es numpy.ndarray, omitiendo conversión para emp_id={emp_id}")
                    embeddings.append(embedding_bytes)
                    nombres_empleados.append(emp_id)
                    continue  # Evitar llamada innecesaria a `convertir_varbinary_a_numpy()`

                # 📌 Verificar si el dato es binario antes de convertirlo
                if isinstance(embedding_bytes, (bytes, bytearray)):
                    embedding = self.convertir_varbinary_a_numpy(embedding_bytes)
                    if embedding is not None:
                        embeddings.append(embedding)
                        nombres_empleados.append(emp_id)
                else:
                    print(f"⚠️ Tipo inesperado de embedding para emp_id={emp_id}: {type(embedding_bytes)}, Valor: {embedding_bytes}")

            except Exception as e:
                print(f" Error al recuperar el embedding para emp_id={emp_id}: {e}")

        return embeddings, nombres_empleados  # ✅ Devuelve correctamente embeddings y IDs



    def convertir_varbinary_a_numpy(self, varbinary_data):
        if varbinary_data is None:
            print("⚠️ El embedding es None, se omite la conversión.")
            return None
        
        # 📌 Verificar que los datos sean binarios antes de procesar
        if not isinstance(varbinary_data, (bytes, bytearray)):
            print(f"⚠️ Tipo inesperado para el embedding: {type(varbinary_data)}, Valor: {varbinary_data}")
            return None  # Evita errores si el dato es un número entero u otro tipo incorrecto

        # 📌 Verificar el tamaño esperado antes de convertir
        expected_size = 512 * 4  # 512 valores float32 * 4 bytes
        if len(varbinary_data) != expected_size:
            print(f"⚠️ Tamaño inesperado de VARBINARY: {len(varbinary_data)} bytes (Esperado: {expected_size})")
            return None
        
        # Deserializar el binario almacenado a numpy array
        try:
            float_values = struct.unpack(f"{len(varbinary_data) // 4}f", varbinary_data)
            return np.array(float_values, dtype=np.float32)
        except Exception as e:
            print(f" Error al convertir VARBINARY a numpy array: {e}")
            return None






    # 📌 Obtener embeddings de un empleado
    def get_embeddings_by_emp_id(self, emp_id):
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT embedding FROM embeddings WHERE emp_id = ?", (emp_id,))
            results = cursor.fetchall()

        return [np.frombuffer(row[0], dtype=np.float32) for row in results]

    # 📌 Eliminar embeddings de un empleado
    def delete_embeddings(self, emp_id):
        with self.connection.cursor() as cursor:
            cursor.execute("DELETE FROM embeddings WHERE emp_id = ?", (emp_id,))
            self.connection.commit()

    # 📌 Normalizar embeddings (para mejorar la comparación de similitud)
    def normalize_embedding(self, embedding):
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    # 📌 Calcular la similitud de coseno entre dos embeddings
    def cosine_similarity(self, embedding1, embedding2):
        return 1 - cosine(embedding1, embedding2)

    # 📌 Obtener umbral dinámico basado en embeddings existentes
    def get_dynamic_threshold(self):
        _, embeddings = self.get_embeddings_arc()

        if len(embeddings) < 5:
            return 0.6  # Umbral base por seguridad

        similarities = [
            self.cosine_similarity(embeddings[i], embeddings[j])
            for i in range(len(embeddings))
            for j in range(i + 1, len(embeddings))
        ]

        mean_sim = np.mean(similarities)
        percentile_90 = np.percentile(similarities, 90)  # Ajuste más preciso

        return max(0.6, min(0.85, percentile_90))

        # 📌 Clusterización de embeddings con DBSCAN dinámico
    def cluster_embeddings(self,embeddings):
        if len(embeddings) < 2:
            return embeddings

        distances = [
            1 - cosine(embeddings[i], embeddings[j])
            for i in range(len(embeddings))
            for j in range(i + 1, len(embeddings))
        ]
        eps = np.percentile(distances, 85)  # Ajuste dinámico

        clustering = DBSCAN(eps=eps, min_samples=2, metric='cosine').fit(embeddings)
        labels = clustering.labels_

        valid_clusters = labels[labels != -1]
        if len(valid_clusters) == 0:
            return embeddings  # Sin clusters válidos, retorna los originales

        main_cluster = np.argmax(np.bincount(valid_clusters))
        return [emb for i, emb in enumerate(embeddings) if labels[i] == main_cluster]


    # 📌 Consolidar embeddings de un empleado (promediar los clusterizados)
    def consolidate_embeddings(self, emp_id):
        embeddings = self.get_embeddings_by_emp_id(emp_id)
        clustered = self.cluster_embeddings(embeddings)

        # ✅ Verificar que hay embeddings antes de calcular la media
        return np.mean(clustered, axis=0) if len(clustered) > 0 else None
    


    def revert_employee_data(self, emp_id):
        try:
            # 1) Eliminar embeddings
            cursor = self.connection.cursor()
            cursor.execute("DELETE FROM embeddings WHERE emp_id = ?", (emp_id,))
            self.connection.commit()

            # 2) Eliminar fotos del empleado
            cursor.execute("DELETE FROM empleado_fotos WHERE emp_id = ?", (emp_id,))
            self.connection.commit()

            # 3) Eliminar empleado
            cursor.execute("DELETE FROM empleados WHERE id = ?", (emp_id,))
            self.connection.commit()

            print(f"✅ Se revirtieron todos los datos para emp_id={emp_id}.")
        except Exception as e:
            print(f"Error en revert_employee_data(emp_id={emp_id}): {e}")





#-------------------------------------------------------------------------------------------------------------------------------------
#ASISTENCIAS
#-------------------------------------------------------------------------------------------------------------------------------------
    def registrar_asistencia(self, emp_id, tipo):
        try:
            fecha_actual = datetime.now().strftime('%Y-%m-%d')
            hora_actual = datetime.now().strftime('%H:%M:%S')

            cursor = self.connection.cursor()
            cursor.execute(
                """
                INSERT INTO asistencias (emp_id, fecha_asis, hora_asis, tipo_asis)
                VALUES (?, ?, ?, ?)
                """
            , (emp_id, fecha_actual, hora_actual, tipo))
            
            self.connection.commit()
            print(f"{Fore.GREEN} Asistencia registrada para empleado ID {emp_id} {Style.RESET_ALL}")
        except pyodbc.Error as e:
            print(f"{Fore.RED} Error al registrar asistencia para empleado ID {emp_id}: {e} {Style.RESET_ALL}")


    def get_asistencias_ci_emp(self, emp_id):
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                SELECT * FROM asistencias WHERE emp_id = ?
                """, (emp_id,))
            asistencias = cursor.fetchall()
            return asistencias
        except pyodbc.Error as e:
            print(f"{Fore.RED} Error al obtener asistencias del empleado ID {emp_id}: {e} {Style.RESET_ALL}")
            return []


    def get_asistencias_all(self):
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                SELECT * FROM asistencias
                """
            )
            embeddings = cursor.fetchall()
            return embeddings
        except pyodbc.Error as e:
            print(f"{Fore.RED} Error al obtener asistencias: {e} {Style.RESET_ALL}")
            return []
        

    def get_asistencias(db, filters=None):
        query = """
        SELECT 
            a.id, 
            e.nombres_emp + ' ' + e.apellidos_emp AS empleado,
            FORMAT(a.fecha_asis, 'yyyy-MM-dd') AS fecha_asis, 
            CAST(a.hora_asis AS TIME) AS hora_asis,  -- Cambia a TIME directamente 
            a.tipo_asis
        FROM asistencias a
        INNER JOIN empleados e ON a.emp_id = e.id
        """

        conditions = []
        params = []

        if filters:
            if 'day' in filters:
                conditions.append("a.fecha_asis = ?")
                params.append(filters['day'])

            if 'month' in filters:
                conditions.append("MONTH(a.fecha_asis) = ?")
                params.append(filters['month'])  

            if 'year' in filters:
                conditions.append("YEAR(a.fecha_asis) = ?")
                params.append(filters['year'])

            if 'name_cedula' in filters:
                conditions.append("(e.nombres_emp LIKE ? OR e.apellidos_emp LIKE ? OR e.ci_emp LIKE ?)")
                search_value = f"%{filters['name_cedula']}%"
                params.extend([search_value, search_value, search_value])

            if 'type' in filters:
                conditions.append("a.tipo_asis = ?")
                params.append(filters['type'])

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY a.fecha_asis DESC, a.hora_asis DESC"

        try:
            cursor = db.connection.cursor()
            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()  # Liberar recursos

            # Convertir la hora a datetime.time
            return [(row[0], row[1], row[2], row[3] if row[3] else None, row[4]) for row in results]

        except pyodbc.Error as e:
            print(f"{Fore.RED} Error al obtener asistencias: {e} {Style.RESET_ALL}")
            return []




        

    def delete_asistencia(self, asistencia_id):
        """Elimina un registro de asistencia."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("DELETE FROM asistencias WHERE id = ?", (asistencia_id,))
            self.connection.commit()
            return cursor.rowcount > 0  # ✅ Retorna True si se eliminó correctamente
        except pyodbc.Error as e:
            print(f"{Fore.RED} Error al eliminar asistencia ID {asistencia_id}: {e} {Style.RESET_ALL}")
            return False



    def update_asistencia(self, asistencia_id, nueva_fecha, nueva_hora):
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                UPDATE asistencias
                SET fecha_asis = ?, hora_asis = ?
                WHERE id = ?
            """, (nueva_fecha, nueva_hora, asistencia_id))
            self.connection.commit()
            return cursor.rowcount > 0 
        except pyodbc.Error as e:
            print(f"{Fore.RED} Error al actualizar asistencia ID {asistencia_id}: {e} {Style.RESET_ALL}")
            return False
        
    def update_asistencia_hora(self, asis_id, nueva_hora):
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "UPDATE asistencias SET hora_asis = CAST(? AS TIME) WHERE id = ?", 
                (str(nueva_hora), asis_id)  
            )
            self.connection.commit()
            print(f"{Fore.GREEN} Hora actualizada correctamente para ID {asis_id}: {nueva_hora} {Style.RESET_ALL}")

        except pyodbc.Error as e:
            print(f"{Fore.RED}Error al actualizar la hora de asistencia ID {asis_id}: {e} {Style.RESET_ALL}")


    def get_recent_attendances(self, limit=6):
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"""
                SELECT TOP {limit} e.id, e.nombres_emp, a.fecha_asis, a.hora_asis, a.tipo_asis
                FROM asistencias a
                JOIN empleados e ON a.emp_id = e.id
                ORDER BY a.fecha_asis DESC, a.hora_asis DESC
            """)
            asistencias = cursor.fetchall()
            return asistencias
        except pyodbc.Error as e:
            print(f"{Fore.RED} Error al obtener asistencias recientes: {e} {Style.RESET_ALL}")
            return []
        


    def validar_registro_asistencia(self, emp_id, nuevo_tipo):
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT TOP 1 tipo_asis
                FROM asistencias
                WHERE emp_id = ?
                ORDER BY fecha_asis DESC, hora_asis DESC
            """, (emp_id,))
            resultado = cursor.fetchone()
            
            if resultado is None:
                return nuevo_tipo == 'entrada'
            
            ultimo_tipo = resultado[0].strip().lower()  # Aseguramos formato uniforme
            nuevo_tipo = nuevo_tipo.strip().lower()
            
            if nuevo_tipo == 'entrada' and ultimo_tipo == 'salida':
                return True
            elif nuevo_tipo == 'salida' and ultimo_tipo == 'entrada':
                return True
            else:
                return False
        except Exception as e:
            print(f"Error al validar asistencia para emp_id {emp_id}: {e}")
            return False
        


    def get_salida_hoy(self, emp_id):
        fecha_hoy = datetime.now().strftime('%Y-%m-%d')
        query = """
            SELECT TOP 1 id
            FROM asistencias
            WHERE emp_id = ? AND fecha_asis = ? AND tipo_asis = 'salida'
            ORDER BY hora_asis
        """
        cursor = self.connection.cursor()
        cursor.execute(query, (emp_id, fecha_hoy))
        row = cursor.fetchone()
        if row:
            return row[0]  # ID del registro
        return None
    
#----------------------------------------------------------------------
#modelel load_camerao2_min
#----------------------------------------------------------------------


    def ya_tiene_entrada_hoy(self, emp_id):
        fecha_hoy = datetime.now().strftime('%Y-%m-%d')
        query = """
            SELECT TOP 1 id
            FROM asistencias
            WHERE emp_id = ? AND fecha_asis = ? AND tipo_asis = 'entrada'
            ORDER BY hora_asis
        """
        cursor = self.connection.cursor()
        cursor.execute(query, (emp_id, fecha_hoy))
        return cursor.fetchone() is not None



    def validar_registro_asistencia_o(self, emp_id, nuevo_tipo):
        fecha_hoy = datetime.now().strftime('%Y-%m-%d')
        nuevo_tipo = nuevo_tipo.lower().strip()

        if nuevo_tipo == 'entrada':
            if self.ya_tiene_entrada_hoy(emp_id):
                print(f"[DB] Empleado {emp_id} ya tiene ENTRADA hoy => se ignora nueva entrada.")
                return False
            return True

        elif nuevo_tipo == 'salida':

            try:
                cursor = self.connection.cursor()
                cursor.execute("""
                    SELECT TOP 1 tipo_asis
                    FROM asistencias
                    WHERE emp_id = ?
                    ORDER BY fecha_asis DESC, hora_asis DESC
                """, (emp_id,))
                fila = cursor.fetchone()
                if not fila:
                    # No había registro => no se puede hacer 'salida' sin 'entrada'
                    return False
                ultimo_tipo = fila[0].strip().lower()
                if ultimo_tipo == 'salida':
                    # No permitir si la última marcación fue salida consecutiva 
                    print(f"[DB] Empleado {emp_id} hizo dos salidas consecutivas => se ignora.")
                    return False

                # Pasó => devolvemos True para proceder a registrar (o actualizar)
                return True

            except Exception as e:
                print(f"[DB] Error en validar_registro_asistencia(salida): {e}")
                return False

        else:
            # Tipo desconocido
            return False
        


    def registrar_asistencia_o(self, emp_id, tipo):
        fecha_hoy = datetime.now().strftime('%Y-%m-%d')
        hora_actual = datetime.now().strftime('%H:%M:%S')
        tipo = tipo.lower().strip()

        try:
            if tipo == 'entrada':
                # Inserta normal
                cursor = self.connection.cursor()
                cursor.execute("""
                    INSERT INTO asistencias (emp_id, fecha_asis, hora_asis, tipo_asis)
                    VALUES (?, ?, ?, ?)
                """, (emp_id, fecha_hoy, hora_actual, tipo))
                self.connection.commit()
                print(f"[DB] Se registró ENTRADA para emp_id={emp_id} en {fecha_hoy} {hora_actual}")

            elif tipo == 'salida':
                # Ver si ya hay una salida hoy:
                id_salida = self.get_salida_hoy(emp_id)
                if id_salida:
                    # Actualizar la hora de ESA salida, para hacerla "última salida"
                    cursor = self.connection.cursor()
                    cursor.execute("""
                        UPDATE asistencias
                        SET hora_asis = ?
                        WHERE id = ?
                    """, (hora_actual, id_salida))
                    self.connection.commit()
                    print(f"[DB] Se ACTUALIZA la SALIDA (id={id_salida}) con nueva hora {hora_actual}")
                else:
                    # No hay => se inserta la primera vez
                    cursor = self.connection.cursor()
                    cursor.execute("""
                        INSERT INTO asistencias (emp_id, fecha_asis, hora_asis, tipo_asis)
                        VALUES (?, ?, ?, ?)
                    """, (emp_id, fecha_hoy, hora_actual, tipo))
                    self.connection.commit()
                    print(f"[DB] Se registró SALIDA (nueva) para emp_id={emp_id} en {fecha_hoy} {hora_actual}")

        except Exception as e:
            print(f"[DB] Error en registrar_asistencia({tipo}): {e}")
            
            
            
            
            
    def validar_registro_asistencia_vino(self, emp_id, nuevo_tipo):
        """
        Lógica simplificada:
        - Si es 'entrada':
            * Sólo es válida si hoy NO existe ya una entrada para este emp_id.
        - Si es 'salida':
            * Se requiere que exista una 'entrada' para hoy.
            * Si no hay 'entrada' hoy, no se registra.
            * El 'salida' como tal se aceptará (la verificación de los 5 minutos se hace en registrar_asistencia_o).
        """
        fecha_hoy = datetime.now().strftime('%Y-%m-%d')
        nuevo_tipo = nuevo_tipo.lower().strip()

        # Verificar si existe una entrada hoy para este empleado
        tiene_entrada_hoy = self.ya_tiene_entrada_hoy(emp_id)

        if nuevo_tipo == 'entrada':
            # Solo permitir si NO hay entrada hoy
            if tiene_entrada_hoy:
                print(f"[DB] Empleado {emp_id} ya tiene ENTRADA hoy => se ignora nueva entrada.")
                return False
            return True

        elif nuevo_tipo == 'salida':
            # Requiere que exista una entrada hoy
            if not tiene_entrada_hoy:
                print(f"[DB] Empleado {emp_id} no tiene ENTRADA hoy => no se registra SALIDA.")
                return False
            # Aceptar la solicitud de salida (el control de los 5 min se hace en registrar_asistencia_o)
            return True

        else:
            # Tipo desconocido
            return False


    def registrar_asistencia_vino(self, emp_id, tipo):
        fecha_hoy = datetime.now().strftime('%Y-%m-%d')
        hora_actual_dt = datetime.now()
        hora_actual_str = hora_actual_dt.strftime('%H:%M:%S')
        tipo = tipo.lower().strip()

        try:
            if tipo == 'entrada':
                # Inserta la primera entrada del día
                cursor = self.connection.cursor()
                cursor.execute("""
                    INSERT INTO asistencias (emp_id, fecha_asis, hora_asis, tipo_asis)
                    VALUES (?, ?, ?, ?)
                """, (emp_id, fecha_hoy, hora_actual_str, tipo))
                self.connection.commit()
                print(f"[DB] Se registró ENTRADA para emp_id={emp_id} el {fecha_hoy} a las {hora_actual_str}")
                return True  # Se registró con éxito

            elif tipo == 'salida':
                id_salida = self.get_salida_hoy(emp_id)
                if id_salida:
                    cursor = self.connection.cursor()
                    cursor.execute("""
                        SELECT hora_asis 
                        FROM asistencias
                        WHERE id = ?
                    """, (id_salida,))
                    row = cursor.fetchone()
                    if row:
                        ultima_hora_salida_time = row[0]  # datetime.time
                        ultima_salida_completa = datetime.combine(date.today(), ultima_hora_salida_time)

                        diff_minutos = (hora_actual_dt - ultima_salida_completa).total_seconds() / 60.0
                        if diff_minutos < COLD_DOWN_ASISTENCIA:
                            print("[DB] Se ignora la nueva SALIDA, no han pasado 5 minutos de la última.")
                            return False  # No se registró
                        else:
                            # Actualizar la hora de ESA salida (id_salida)
                            cursor.execute("""
                                UPDATE asistencias
                                SET hora_asis = ?
                                WHERE id = ?
                            """, (hora_actual_str, id_salida))
                            self.connection.commit()
                            print(f"[DB] Se ACTUALIZA la SALIDA (id={id_salida}) con nueva hora {hora_actual_str}")
                            return True  # Se actualizó con éxito
                else:
                    # No hay salida hoy => se inserta la primera vez
                    cursor = self.connection.cursor()
                    cursor.execute("""
                        INSERT INTO asistencias (emp_id, fecha_asis, hora_asis, tipo_asis)
                        VALUES (?, ?, ?, ?)
                    """, (emp_id, fecha_hoy, hora_actual_str, tipo))
                    self.connection.commit()
                    print(f"[DB] Se registró SALIDA (nueva) para emp_id={emp_id} el {fecha_hoy} a las {hora_actual_str}")
                    return True  # Se insertó con éxito

        except Exception as e:
            print(f"[DB] Error en registrar_asistencia_o({tipo}): {e}")

        # Si algo falla, retornar False
        return False





#-------------------------------------------------------------------------------------------------------------------------------------
#USUARIOS
#-------------------------------------------------------------------------------------------------------------------------------------
    def registrar_usuario(self, usuario, password, rol="GENERAL"):
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                INSERT INTO usuarios (usuario, password, rol)
                VALUES (?, ?, ?)
                """, 
                (usuario, password, rol))
            
            self.connection.commit()
            print(f"{Fore.GREEN} Usuario {usuario} registrado correctamente con rol {rol} {Style.RESET_ALL}")
        except pyodbc.Error as e:
            print(f"{Fore.RED} Error al registrar usuario {usuario}: {e} {Style.RESET_ALL}")


    def registrar_usuario(self, usuario, password, rol="usuario"):
        try:
            cursor = self.connection.cursor()
            password_hash = ph.hash(password)  # Hashear la contraseña con Argon2
            
            cursor.execute("""
                INSERT INTO usuarios (usuario, password_hash, rol) 
                VALUES (?, ?, ?)
            """, (usuario, password_hash, rol))
            
            self.connection.commit()
            print(f"{Fore.GREEN} ✅ Usuario '{usuario}' registrado correctamente con rol '{rol}' {Style.RESET_ALL}")
        except pyodbc.IntegrityError:
            print(f"{Fore.RED}  Error: El usuario '{usuario}' ya existe. {Style.RESET_ALL}")
        except pyodbc.Error as e:
            print(f"{Fore.RED}  Error al registrar usuario '{usuario}': {e} {Style.RESET_ALL}")



    def autenticar_usuario(self, usuario, password):
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT password_hash, rol FROM usuarios WHERE usuario = ?", (usuario,))
            resultado = cursor.fetchone()

            if resultado is None:
                print(f"{Fore.RED}  Usuario '{usuario}' no encontrado. {Style.RESET_ALL}")
                return None  # Devuelve None si el usuario no existe

            password_hash, rol = resultado

            if not password_hash:
                print(f"{Fore.RED}  La contraseña almacenada es inválida o está vacía. {Style.RESET_ALL}")
                return None

            if ph.verify(password_hash, password):
                print(f"{Fore.GREEN}  Inicio de sesión exitoso. Rol: {rol} {Style.RESET_ALL}")
                return rol  # Retorna el rol del usuario si la autenticación es exitosa
            else:
                print(f"{Fore.RED}  La contraseña ingresada es incorrecta. {Style.RESET_ALL}")
                return None

        except argon2.exceptions.VerifyMismatchError:
            print(f"{Fore.RED}  Error: La contraseña no coincide con el hash almacenado. {Style.RESET_ALL}")
            return None
        except Exception as e:
            print(f"{Fore.RED}  Error en la autenticación del usuario '{usuario}': {e} {Style.RESET_ALL}")
            return None

    def existen_usuarios(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM usuarios")
        return cursor.fetchone()[0] > 0
    
    def cambiar_estado_empleado(self, emp_id, nuevo_estado):
        try:
            cursor = self.connection.cursor()

            if not emp_id:
                print(f"{Fore.RED} Error: ID del empleado no válido. {Style.RESET_ALL}")
                return False

            cursor.execute("UPDATE empleados SET estado_emp = ? WHERE id = ?", (nuevo_estado, emp_id))
            self.connection.commit()
            
            estado_texto = "activado" if nuevo_estado == 1 else "desactivado"
            print(f"{Fore.GREEN} Empleado ID {emp_id} {estado_texto} correctamente. {Style.RESET_ALL}")
            return True

        except pyodbc.Error as e:
            print(f"{Fore.RED} Error al cambiar estado del empleado ID {emp_id}: {e} {Style.RESET_ALL}")
            return False




    def eliminar_usuario(self, usuario):
        try:
            cursor = self.connection.cursor()
            cursor.execute("DELETE FROM usuarios WHERE usuario = ?", (usuario,))
            self.connection.commit()
            return cursor.rowcount > 0  # Retorna True si se eliminó correctamente
        except pyodbc.Error as e:
            print(f"Error al eliminar usuario {usuario}: {e}")
            return False

    # Método para obtener la lista de usuarios
    def obtener_usuarios(self):
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT usuario FROM usuarios")
            usuarios = [row[0] for row in cursor.fetchall()]
            return usuarios
        except pyodbc.Error as e:
            print(f"Error al obtener la lista de usuarios: {e}")
            return []


        

   
            

#-------------------------------------------------------------------------------------------------------------------------------------
#CARAS DE DESCONOCIDOS
#-------------------------------------------------------------------------------------------------------------------------------------
    def save_unknown_face(self, image_data):
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                INSERT INTO caras_desconocidas (imagen)
                VALUES (?)
                """, 
                (pyodbc.Binary(image_data),)  # ✅ Asegurar que se almacena como binario
            )
            self.connection.commit()
            print("✅ Cara desconocida guardada correctamente.")
        except pyodbc.Error as e:
            print(f" Error al guardar cara desconocida: {e}")



    def get_unknown_faces(self):
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                SELECT id, fecha, imagen FROM caras_desconocidas
                """)
            caras = cursor.fetchall()
            return caras
        except pyodbc.Error as e:
            print(f"{Fore.GREEN} Error al obtener caras desconocidas: {e} {Style.RESET_ALL}")
            return []

    def delete_unknown_face(self, face_id):
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                DELETE FROM caras_desconocidas WHERE id = ?
                """, 
                (face_id,))
            self.connection.commit()
            print(f"{Fore.GREEN} Cara desconocida ID {face_id} eliminada correctamente {Style.RESET_ALL}")
        except pyodbc.Error as e:
            print(f"{Fore.RED} Error al eliminar cara desconocida ID {face_id}: {e} {Style.RESET_ALL}")


#-------------------------------------------------------------------------------------------------------------------------------------
#FUNCIONES
#-------------------------------------------------------------------------------------------------------------------------------------



    def generate_report_asistencias(self, filters, file_path):
        try:
            asistencias = self.get_asistencias(filters)

            if not asistencias:
                print(f"{Fore.YELLOW} No hay asistencias para el filtro seleccionado {Style.RESET_ALL}")
                return False

            workbook = xlsxwriter.Workbook(file_path)
            worksheet = workbook.add_worksheet("Asistencias")

            headers = ["ID", "Empleado", "Fecha", "Hora", "Tipo"]
            worksheet.write_row(0, 0, headers)

            date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})
            time_format = workbook.add_format({'num_format': 'hh:mm:ss AM/PM'})

            for row, asistencia in enumerate(asistencias, start=1):
                id_, empleado, fecha, hora, tipo = asistencia

                # ✅ Convertir fecha a string en formato `YYYY-MM-DD`
                if isinstance(fecha, datetime):
                    fecha_convertida = fecha.strftime('%Y-%m-%d')
                elif isinstance(fecha, str):
                    fecha_convertida = fecha  # Si ya es string, no necesita conversión
                else:
                    fecha_convertida = str(fecha)  # Convierte cualquier otro formato a string

                # ✅ Convertir hora a string en formato `HH:MM:SS`
                if isinstance(hora, time):
                    hora_convertida = hora.strftime('%H:%M:%S')
                elif isinstance(hora, float):  
                    total_seconds = int(hora * 86400)  # Convierte la fracción del día a segundos
                    hours = total_seconds // 3600
                    minutes = (total_seconds % 3600) // 60
                    seconds = total_seconds % 60
                    hora_convertida = f"{hours:02}:{minutes:02}:{seconds:02}"
                else:
                    hora_convertida = str(hora)

                worksheet.write(row, 0, id_)
                worksheet.write(row, 1, empleado)
                worksheet.write(row, 2, fecha_convertida, date_format)
                worksheet.write(row, 3, hora_convertida, time_format)
                worksheet.write(row, 4, tipo)

            workbook.close()
            print(f"{Fore.GREEN} Reporte generado correctamente en {file_path} {Style.RESET_ALL}")
            return True
        except Exception as e:
            print(f"{Fore.RED} Error al generar el reporte: {e} {Style.RESET_ALL}")
            return False



        
        

    def get_name_folder(db, emp_id):
        try:
            cursor = db.connection.cursor()
            cursor.execute(
                "SELECT LEFT(apellidos_emp, 5) + '_' + LEFT(nombres_emp, 5) + CAST(id AS VARCHAR) AS nombre_carpeta FROM empleados WHERE id = ?;",
                (emp_id,)
            )
            name_folder = cursor.fetchone()
            return name_folder[0] if name_folder else None  # Devuelve solo el nombre de la carpeta o None si no hay resultados
        except pyodbc.Error as e:
            print(f"{Fore.RED} Error al obtener nombre de carpeta: {e} {Style.RESET_ALL}")
            return None

        
    def add_ruta_carpeta_emp(self, emp_id, ruta_carpeta):
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO empleado_fotos (emp_id, ruta_carpeta)
                VALUES (?, ?)
            """, (emp_id, ruta_carpeta))
            self.connection.commit()
            print(f"{Fore.GREEN} Ruta de carpeta guardada para empleado ID {emp_id} {Style.RESET_ALL}")
        except pyodbc.Error as e:
            print(f"{Fore.RED} Error al guardar la ruta de carpeta del empleado ID {emp_id}: {e} {Style.RESET_ALL}")

    def delete_rutas_por_empleado(self, emp_id):
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                DELETE FROM empleado_fotos WHERE emp_id = ?
            """, (emp_id,))
            self.connection.commit()
            print(f"{Fore.GREEN} Ruta de carpeta eliminada para empleado ID {emp_id} {Style.RESET_ALL}")
        except pyodbc.Error as e:
            print(f"{Fore.RED} Error al eliminar la ruta de carpeta del empleado ID {emp_id}: {e} {Style.RESET_ALL}")


    def calculate_hours(self, emp_id):
        """Calcula las horas y minutos trabajados correctamente."""
        try:
            query = """
            WITH Pairs AS (
                SELECT
                    emp_id,
                    CAST(fecha_asis AS DATETIME) + CAST(hora_asis AS DATETIME) AS entrada,
                    LEAD(CAST(fecha_asis AS DATETIME) + CAST(hora_asis AS DATETIME)) 
                        OVER (PARTITION BY emp_id ORDER BY fecha_asis, hora_asis) AS salida,
                    tipo_asis
                FROM asistencias
                WHERE tipo_asis IN ('entrada', 'salida')
            ),
            FilteredPairs AS (
                SELECT
                    emp_id,
                    entrada,
                    salida,
                    DATEDIFF(MINUTE, entrada, salida) AS minutos_trabajados
                FROM Pairs
                WHERE tipo_asis = 'entrada' AND salida IS NOT NULL
                    AND MONTH(entrada) = MONTH(GETDATE()) 
                    AND YEAR(entrada) = YEAR(GETDATE())
            )
            SELECT
                ISNULL(SUM(minutos_trabajados) / 60, 0) AS horas,
                ISNULL(SUM(minutos_trabajados) % 60, 0) AS minutos
            FROM FilteredPairs
            WHERE emp_id = ?;
            """
            cursor = self.connection.cursor()
            cursor.execute(query, (emp_id,))
            result = cursor.fetchone()

            if result:
                horas, minutos = result
                return f"{int(horas)}h {int(minutos)}m"  # Formato `Xh Ym`
            else:
                return "0h 0m"  # Si no hay registros

        except Exception as e:
            print(f"{Fore.RED} Error al calcular horas trabajadas para emp_id {emp_id}: {e}{Style.RESET_ALL}")
            return "Error"
        

    def folder_validate(self, emp_id):
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT ruta_carpeta FROM empleado_fotos WHERE emp_id = ?",
                (emp_id,)
            )
            validate = cursor.fetchone()
            return validate
        except pyodbc.Error as e:
            print(f"{Fore.RED} Error de comprovacion: {e} {Style.RESET_ALL}")
            return None


    def get_unknown_faces(self, date_filter=None):
        try:
            cursor = self.connection.cursor()
            if date_filter:
                query = "SELECT id, fecha, imagen FROM caras_desconocidas WHERE fecha = ?"
                cursor.execute(query, (date_filter,))
            else:
                query = "SELECT id, fecha, imagen FROM caras_desconocidas"
                cursor.execute(query)

            caras = cursor.fetchall()
            return caras
        except pyodbc.Error as e:
            print(f"Error al obtener caras desconocidas: {e}")
            return []



        
    def delete_unknown_face(self, face_id):
        try:
            cursor = self.connection.cursor()
            cursor.execute("DELETE FROM caras_desconocidas WHERE id = ?", (face_id,))
            self.connection.commit()
            print(f"{Fore.GREEN} Rostro ID {face_id} eliminado correctamente. {Style.RESET_ALL}")
        except pyodbc.Error as e:
            print(f"{Fore.RED} Error al eliminar rostro ID {face_id}: {e} {Style.RESET_ALL}")

    def get_unknown_face_by_id(self, face_id):
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT imagen FROM caras_desconocidas WHERE id = ?", (face_id,))
            resultado = cursor.fetchone()
            return resultado[0] if resultado else None
        except pyodbc.Error as e:
            print(f"{Fore.RED} Error al obtener rostro ID {face_id}: {e} {Style.RESET_ALL}")
            return None
        




        












   # 📌 Crear y almacenar un embedding de prueba (NORMALIZADO CORRECTAMENTE)
    def store_test_embedding(self, emp_id):
        embedding = np.random.rand(512).astype(np.float32) * 2 - 1  # Generar valores en [-1, 1]
        embedding = embedding / np.linalg.norm(embedding)  # Normalizar

        embedding_bytes = embedding.tobytes()  # Convertir a binario

        cursor = self.connection.cursor()
        cursor.execute("INSERT INTO embeddings (emp_id, embedding) VALUES (?, ?)", (emp_id, embedding_bytes))
        self.connection.commit()
        print(f"✅ Embedding de prueba almacenado para emp_id={emp_id}")

    # 📌 Recuperar y comprobar el embedding
    def check_embedding(self, emp_id):
        cursor = self.connection.cursor()
        cursor.execute("SELECT embedding FROM embeddings WHERE emp_id = ?", (emp_id,))
        row = cursor.fetchone()

        if row:
            recovered_embedding = np.frombuffer(row[0], dtype=np.float32)  # Convertir de binario a NumPy

            # 📌 Verificaciones
            print(f"✅ Embedding recuperado para emp_id={emp_id}:")
            print(f"Dimensión: {len(recovered_embedding)}")  # Debe ser 512
            print(f"Norma: {np.linalg.norm(recovered_embedding):.6f}")  # Debe ser ~1.0
            print(f"Rango -> Mínimo: {np.min(recovered_embedding):.6f}, Máximo: {np.max(recovered_embedding):.6f}")
            print(f"Primeros 10 valores: {recovered_embedding[:10]}")
            
        else:
            print(f" No se encontró embedding para emp_id={emp_id}")



    def auto_generate_employee_report(self, period, file_path):
        try:
            filters = {}

            # Aplicar filtro según el período seleccionado
            if period == "diario":
                filters['day'] = datetime.now().strftime("%Y-%m-%d")
            elif period == "semanal":
                start_week = datetime.now() - timedelta(days=datetime.now().weekday())
                filters['day'] = start_week.strftime("%Y-%m-%d")
            elif period == "mensual":
                filters['month'] = datetime.now().month
                filters['year'] = datetime.now().year
            elif period == "anual":
                filters['year'] = datetime.now().year
            else:
                print(f"Período no válido: {period}")
                return False

            asistencias = self.get_asistencias(filters)

            if not asistencias:
                print(f"No hay asistencias para el período {period}")
                return False

            workbook = xlsxwriter.Workbook(file_path)
            worksheet = workbook.add_worksheet("Reporte de Asistencias")

            headers = ["Empleado", "Fecha", "Hora Entrada", "Hora Salida", "Horas Trabajadas"]
            worksheet.write_row(0, 0, headers)

            date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})
            time_format = workbook.add_format({'num_format': 'hh:mm:ss AM/PM'})

            data = {}
            for id_, empleado, fecha, hora, tipo in asistencias:
                if empleado not in data:
                    data[empleado] = {'entrada': None, 'salida': None, 'total_minutos': 0}

                if tipo == "entrada":
                    data[empleado]['entrada'] = hora
                elif tipo == "salida" and data[empleado]['entrada']:
                    entrada_dt = datetime.strptime(data[empleado]['entrada'], "%H:%M:%S")
                    salida_dt = datetime.strptime(hora, "%H:%M:%S")
                    minutos_trabajados = (salida_dt - entrada_dt).seconds // 60
                    data[empleado]['salida'] = hora
                    data[empleado]['total_minutos'] += minutos_trabajados
                    data[empleado]['entrada'] = None  # Reset para próximas entradas

            row = 1
            for empleado, info in data.items():
                horas_trabajadas = f"{info['total_minutos'] // 60}h {info['total_minutos'] % 60}m"
                worksheet.write(row, 0, empleado)
                worksheet.write(row, 1, datetime.now().strftime("%Y-%m-%d"), date_format)
                worksheet.write(row, 2, info['entrada'] or "-", time_format)
                worksheet.write(row, 3, info['salida'] or "-", time_format)
                worksheet.write(row, 4, horas_trabajadas)
                row += 1

            workbook.close()
            print(f"Reporte generado correctamente en {file_path}")
            return True

        except Exception as e:
            print(f"Error al generar el reporte: {e}")
            return False
        

    def auto_get_asistencias(self, filters=None):
        query = """
        SELECT 
            a.id, 
            e.nombres_emp + ' ' + e.apellidos_emp AS empleado,
            FORMAT(a.fecha_asis, 'yyyy-MM-dd') AS fecha_asis, 
            CAST(a.hora_asis AS TIME) AS hora_asis,  
            a.tipo_asis
        FROM asistencias a
        INNER JOIN empleados e ON a.emp_id = e.id
        """

        conditions = []
        params = []

        if filters:
            if 'day' in filters:
                conditions.append("a.fecha_asis = ?")
                params.append(filters['day'])

            if 'month' in filters:
                conditions.append("MONTH(a.fecha_asis) = ?")
                params.append(filters['month'])  

            if 'year' in filters:
                conditions.append("YEAR(a.fecha_asis) = ?")
                params.append(filters['year'])

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY a.fecha_asis DESC, a.hora_asis DESC"

        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()

            return [(row[0], row[1], row[2], row[3], row[4]) for row in results]

        except Exception as e:
            print(f"Error al obtener asistencias: {e}")
            return []



















if __name__ == "__main__":
    db = DatabaseManager()

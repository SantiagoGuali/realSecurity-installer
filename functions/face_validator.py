from scipy.spatial.distance import euclidean

class GestorPersistenciaRostros:
    def __init__(self, max_distancia=50, frames_estabilidad=5):
        # Almacena internamente: {centro: (nombre, confianza, frames_detectado)}
        self.rostros_confirmados = {}
        self.max_distancia = max_distancia
        self.frames_estabilidad = frames_estabilidad

    def actualizar_rostros(self, rostros_detectados):
        """
        'rostros_detectados' es un dict: {centro: (nombre, confianza)}.

        Internamente seguiremos usando (nombre, confianza, frames_detectado).
        Finalmente retornamos un dict {centro: (nombre, confianza)} para que
        fuera de esta clase se desempaquen solo 2 valores.
        """
        nuevos_rostros = {}

        for centro, (nombre, confianza) in rostros_detectados.items():
            # Buscar si ya existe un rostro cercano
            rostro_existente = None
            for prev_centro in self.rostros_confirmados:
                # Revisar si está dentro de max_distancia
                if euclidean(centro, prev_centro) < self.max_distancia:
                    rostro_existente = prev_centro
                    break

            if rostro_existente:
                # Recuperar lo que había antes
                nombre_prev, confianza_prev, frames_detectado = self.rostros_confirmados[rostro_existente]
                # Calculamos la nueva confianza mezclando la previa con la actual
                nueva_confianza = (confianza_prev * 0.7 + confianza * 0.3)
                nuevos_rostros[centro] = (nombre, nueva_confianza, frames_detectado + 1)
            else:
                # Rostro nuevo
                nuevos_rostros[centro] = (nombre, confianza, 1)

        # Conservamos solo los que no superan frames_estabilidad
        self.rostros_confirmados = {
            c: nuevos_rostros[c]
            for c in nuevos_rostros
            if nuevos_rostros[c][2] < self.frames_estabilidad
        }

        # Ahora construimos el dict FINAL con SOLO 2 valores (nombre, confianza)
        # para que en tu bucle principal no haya error al desempaquetar:
        rostros_final = {}
        for c, (n, conf, fr) in self.rostros_confirmados.items():
            rostros_final[c] = (n, conf)

        return rostros_final

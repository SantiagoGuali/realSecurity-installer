from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

# Inicializar Argon2 con configuraciones seguras
ph = PasswordHasher(
    time_cost=2,     # Tiempo de cómputo (seguridad vs. rendimiento)
    memory_cost=102400,  # Uso de memoria en KB (102MB)
    parallelism=8,    # Número de hilos paralelos
)

def encriptar_pass(password):
    """
    Hashea una contraseña utilizando Argon2.
    """
    return ph.hash(password)

def verificar_pass(contraseña, hash_almacenado):
    """
    Verifica si la contraseña ingresada coincide con el hash almacenado.
    """
    try:
        return ph.verify(hash_almacenado, contraseña)
    except VerifyMismatchError:
        return False




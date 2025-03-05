-- empleados
IF OBJECT_ID('empleados', 'U') IS NULL
    CREATE TABLE empleados (
        id INT IDENTITY(1,1) PRIMARY KEY,
        ci_emp NVARCHAR(50) NOT NULL UNIQUE,
        apellidos_emp NVARCHAR(100) NOT NULL,
        nombres_emp NVARCHAR(100) NOT NULL,
        mail_emp NVARCHAR(255),
        phone_emp NVARCHAR(20) NOT NULL,
        fechaN_emp DATE,
        genero_emp NVARCHAR(20),
        area_emp NVARCHAR(100),
        estado_emp BIT DEFAULT 1
    );

-- rutas de empleado
IF OBJECT_ID('empleado_fotos', 'U') IS NULL
    CREATE TABLE empleado_fotos (
        id INT IDENTITY(1,1) PRIMARY KEY,
        emp_id INT,
        ruta_carpeta NVARCHAR(255) NOT NULL,
        CONSTRAINT FK_empleado_fotos FOREIGN KEY (emp_id) REFERENCES empleados(id) ON DELETE NO ACTION
    );

-- embeddings
IF OBJECT_ID('embeddings', 'U') IS NULL
    CREATE TABLE embeddings (
        id INT IDENTITY(1,1) PRIMARY KEY,
        emp_id INT,
        embedding VARBINARY(MAX) NOT NULL, 
        estado_emb BIT DEFAULT 1,
        CONSTRAINT FK_embeddings FOREIGN KEY (emp_id) REFERENCES empleados(id) ON DELETE NO ACTION
    );

-- asistencias
IF OBJECT_ID('asistencias', 'U') IS NULL
    CREATE TABLE asistencias (
        id INT IDENTITY(1,1) PRIMARY KEY,
        emp_id INT,
        fecha_asis DATE NOT NULL,
        hora_asis TIME NOT NULL,
        tipo_asis VARCHAR(10) NOT NULL, 
        estado_asis VARCHAR(20) DEFAULT 'Presente',
        CONSTRAINT FK_asistencias FOREIGN KEY (emp_id) REFERENCES empleados(id) ON DELETE NO ACTION
    );

-- usuarios
IF OBJECT_ID('usuarios', 'U') IS NULL
CREATE TABLE usuarios (
    id INT IDENTITY PRIMARY KEY,
    usuario NVARCHAR(50) UNIQUE NOT NULL,
    password_hash NVARCHAR(255) NOT NULL,
    rol NVARCHAR(25),
);


-- desconocidos
IF OBJECT_ID('caras_desconocidas', 'U') IS NULL
    CREATE TABLE caras_desconocidas (
        id INT IDENTITY(1,1) PRIMARY KEY,
        fecha DATETIME DEFAULT GETDATE(),
        imagen VARBINARY(MAX) NOT NULL
    );

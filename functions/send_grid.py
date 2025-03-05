import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import base64
import certifi
from dotenv import load_dotenv
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
from utils.settings_controller import EMAIL_USER

# Forzar el uso de certificados actualizados
os.environ['SSL_CERT_FILE'] = certifi.where()
load_dotenv()

def enviar_alerta_correo(face_image):
    success, buffer = cv2.imencode('.jpg', face_image)
    if not success:
        print("Error al codificar la imagen del rostro.")
        return False

    image_data = buffer.tobytes()
    encoded_file = base64.b64encode(image_data).decode()

    # Crear el objeto Attachment para la imagen
    attachment = Attachment(
        FileContent(encoded_file),
        FileName("rostro_detectado.jpg"),
        FileType("image/jpeg"),
        Disposition("attachment")
    )

    # Crear el mensaje de alerta con contenido HTML estilizado y versión de texto plano
    message = Mail(
        from_email='santiago.gualichico1903@utc.edu.ec',  # Asegúrate de que esté verificado en SendGrid
        to_emails=EMAIL_USER,
        subject='Alerta de Seguridad RealSecurity - Rostro Detectado',
        html_content="""
        <html>
          <head>
            <meta charset="UTF-8">
            <style>
              body { font-family: Arial, sans-serif; background-color: #f4f4f4; padding: 20px; }
              .container { background-color: #ffffff; padding: 20px; border-radius: 5px; max-width: 600px; margin: auto; }
              .header { text-align: center; padding-bottom: 20px; }
              .header h1 { color: #d9534f; margin: 0; }
              .content { font-size: 16px; color: #333333; line-height: 1.5; }
              .footer { margin-top: 20px; font-size: 12px; color: #777777; text-align: center; }
              .button { display: inline-block; margin-top: 20px; padding: 10px 20px; background-color: #0275d8; color: #fff; text-decoration: none; border-radius: 3px; }
            </style>
          </head>
          <body>
            <div class="container">
              <div class="header">
                <h1>Alerta de Seguridad</h1>
              </div>
              <div class="content">
                <p>Se ha detectado un rostro desconocido en el sistema <strong>RealSecurity</strong>.</p>
                <p>Revisa el archivo adjunto para ver la imagen capturada.</p>
              </div>
              <div class="footer">
                <p>Este mensaje ha sido enviado por RealSecurity.</p>
              </div>
            </div>
          </body>
        </html>
        """,
        plain_text_content="""
Alerta de Seguridad

Se ha detectado un rostro desconocido en el sistema RealSecurity.
Revisa el archivo adjunto para ver la imagen capturada.
        """
    )

    # Agregar el adjunto al mensaje
    message.attachment = attachment

    # Enviar el correo usando SendGrid
    try:
        sg = SendGridAPIClient(os.getenv("API_KEY_SENDGRID"))
        response = sg.send(message)
        print("Código de respuesta:", response.status_code)
        return True
    except Exception as e:
        print("Error al enviar el correo:", e)
        return False

# Ejemplo de uso:
if __name__ == "__main__":
    # Supón que 'face_image' es la imagen del rostro detectado (un array NumPy obtenido, por ejemplo, de cv2)
    face_image = cv2.imread("ruta/a/tu/imagen_local.jpg")  # Reemplaza por la imagen capturada en tiempo real
    if face_image is not None:
        enviar_alerta_correo(face_image)
    else:
        print("No se pudo cargar la imagen.")




def enviar_reporte_por_correo(file_path, recipient_email):
    try:
        with open(file_path, "rb") as f:
            file_data = f.read()

        encoded_file = base64.b64encode(file_data).decode()

        # Crear el objeto Attachment para el archivo
        attachment = Attachment(
            FileContent(encoded_file),
            FileName(os.path.basename(file_path)),
            FileType("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            Disposition("attachment")
        )

        message = Mail(
            from_email="santiago.gualichico1903@utc.edu.ec",  # Asegúrate de que esté verificado en SendGrid
            to_emails=recipient_email,
            subject="Reporte de Asistencias - RealSecurity",
            html_content=f"""
            <html>
                <body>
                    <p>Adjunto encontrarás el reporte de asistencias.</p>
                    <p>Gracias por usar <strong>RealSecurity</strong>.</p>
                </body>
            </html>
            """,
            plain_text_content="Adjunto encontrarás el reporte de asistencias. Gracias por usar RealSecurity."
        )

        message.attachment = attachment

        sg = SendGridAPIClient(os.getenv("API_KEY_SENDGRID"))
        response = sg.send(message)
        print(f"Reporte enviado correctamente a {recipient_email}. Código de respuesta: {response.status_code}")

    except Exception as e:
        print(f"Error al enviar el correo: {e}")




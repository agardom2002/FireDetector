import os
from email .message import EmailMessage
import ssl
import smtplib
import streamlit as st

def enviar_correo(email_reciver):
    email_sender = "firedetector.iabd@gmail.com"
    password = st.secrets["PASSWORD"]
    subject = "¡¡¡ALERTA FUEGO!!!!"
    body = """
    ¡¡¡FUEGO EN LAS INMEDIACIONES!!!!
    Su dispositivo FireDetector ha detectado fuego en las inmediaciones de su cámara.
    """
    em = EmailMessage()
    em["From"] = email_sender
    em["To"] = email_reciver
    em["Subject"] = subject
    em.set_content(body)

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
        smtp.login(email_sender, password)
        smtp.sendmail(email_sender, email_reciver, em.as_string())

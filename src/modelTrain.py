from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch

# Cargar el modelo y el tokenizador previamente guardados
model = RobertaForSequenceClassification.from_pretrained('./model_test_1')
tokenizer = RobertaTokenizer.from_pretrained('./model_test_1')

# Definir el mapa de etiquetas
label_map = {
    0: "Autenticación",
    1: "Autorización",
    2: "Encriptación",
    3: "Validación de entrada",
    4: "Registro y monitoreo"
}

# Definir la función de predicción
def predict_tactics(code_snippet):
    inputs = tokenizer(code_snippet, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

    # Realizar la predicción
    with torch.no_grad():  # No necesitamos gradientes para predicciones
        outputs = model(**inputs)
        logits = outputs.logits

    # Obtener la predicción de la clase
    predicted_class_idx = torch.argmax(logits, dim=-1).item()

    # Obtener la clase predicha usando el label_map
    predicted_tactic = label_map[predicted_class_idx]
    return predicted_tactic

# Probar con un fragmento de código
security_code = """
class Authenticator:
    def __init__(self, password):
        self.password = password

    def validate_password(self, input_password):
        if input_password == self.password:
            return True
        return False

auth = Authenticator("secure_password")
input_password = "secure_password"
print(auth.validate_password(input_password))  # Predicción relacionada con autenticación
"""

# Probar con un fragmento de código no relacionado con la seguridad
non_security_code = """
for i in range(1, 11):
    print(i)  # Simple loop, sin relación con tácticas de seguridad
"""

predicted_tactic_security = predict_tactics(security_code)
predicted_tactic_non_security = predict_tactics(non_security_code)

print(f"Táctica de seguridad predicha para el código de seguridad: {predicted_tactic_security}")
print(f"Táctica de seguridad predicha para el código no relacionado: {predicted_tactic_non_security}")

from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch

# Cargar el modelo y el tokenizador previamente entrenados con la data reducida
model = RobertaForSequenceClassification.from_pretrained('./model_test_1')
tokenizer = RobertaTokenizer.from_pretrained('./model_test_1')

# Nuevo label_map con solo las clases presentes en la data reducida
label_map = {
    0: "unrelated",
    1: "3des"
}

# Definir la función de predicción
def predict_class(code_snippet):
    inputs = tokenizer(code_snippet, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

    # Realizar la predicción
    with torch.no_grad():  # No necesitamos gradientes para predicciones
        outputs = model(**inputs)
        logits = outputs.logits

    # Obtener la predicción de la clase
    predicted_class_idx = torch.argmax(logits, dim=-1).item()

    # Obtener la clase predicha usando el label_map
    predicted_label = label_map[predicted_class_idx]
    return predicted_label

# Probar con un fragmento de código relacionado con "3des"
code_3des = """
from Crypto.Cipher import DES3
key = b'Sixteen byte key'
cipher = DES3.new(key, DES3.MODE_ECB)
plaintext = b'Attack at dawn'
ciphertext = cipher.encrypt(plaintext)
"""

# Probar con un código sin relación ("unrelated")
code_unrelated = """
for i in range(1, 11):
    print(i)  # Simple loop, sin relación con criptografía
"""

# Realizar las predicciones
predicted_class_3des = predict_class(code_3des)
predicted_class_unrelated = predict_class(code_unrelated)

print(f"Predicción para código 3DES: {predicted_class_3des}")
print(f"Predicción para código no relacionado: {predicted_class_unrelated}")

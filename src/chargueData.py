import os
import pandas as pd
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
import torch
import numpy as np
from datasets import Dataset

# Ruta de la carpeta donde se encuentran los archivos CSV
folder_path = "src/data"

# Lista para almacenar los DataFrames
dataframes = []

# Recorrer los archivos y cargar los CSV
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        dataframes.append(df)

# Concatenar todos los DataFrames en uno solo
all_data = pd.concat(dataframes, ignore_index=True)

# Preprocesamiento de datos
all_data = all_data.drop(columns=['id'])  # Eliminar columna id
all_data['class'] = all_data['class'].astype(str)  # Convertir a string

# Cargar el tokenizador de CodeBERT
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Mapear las categorías de 'class' a números
label_map = {label: idx for idx, label in enumerate(all_data['class'].unique())}
all_data['label'] = all_data['class'].map(label_map)

# Obtener las clases únicas de la columna 'class'
unique_classes = all_data['class'].unique()

# Crear el label_map que asigna un índice a cada clase
label_map = {label: idx for idx, label in enumerate(unique_classes)}

# Mostrar el label_map para ver las clases con sus índices
print(label_map)


# Separar los datos en entrenamiento y prueba
train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)

# Función para tokenizar los textos
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

# Convertir los DataFrames en un Dataset de Hugging Face
train_data = Dataset.from_pandas(train_data)
test_data = Dataset.from_pandas(test_data)

# Tokenizar los datos
train_data = train_data.map(tokenize_function, batched=True)
test_data = test_data.map(tokenize_function, batched=True)



# Calcular los pesos de las clases para manejar el desbalance
# class_weights = compute_class_weight(
#     'balanced', 
#     classes=np.unique(list(train_data['label'])),  # Usar los valores de 'label' del Dataset
#     y=list(train_data['label'])  # Usar los valores de 'label' del Dataset
# )

# # Convertir los pesos de clase a tensores de PyTorch
# class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# # Cargar el modelo CodeBERT
# model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=len(np.unique(list(train_data['label']))))

# # Definir los argumentos de entrenamiento
# training_args = TrainingArguments(
#     output_dir='./results',          # Directorio de salida
#     num_train_epochs=3,              # Número de épocas
#     per_device_train_batch_size=16,  # Tamaño del batch en entrenamiento
#     per_device_eval_batch_size=64,   # Tamaño del batch en evaluación
#     warmup_steps=500,                # Pasos de calentamiento
#     weight_decay=0.01,               # Decaimiento de peso
#     logging_dir='./logs',            # Directorio de logs
#     logging_steps=10,
#     evaluation_strategy="epoch",     # Evaluar por cada época
#     save_strategy="epoch",
#     load_best_model_at_end=True      # Cargar el mejor modelo al final
# )

# # Función de pérdida con pesos de clase
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_data,
#     eval_dataset=test_data,
#     compute_metrics=None,  # Si no quieres métricas personalizadas, puedes dejarlo como None
# )

# # Entrenar el modelo
# trainer.train()

# # Evaluar el modelo
# results = trainer.evaluate()
# print(results)

# model.save_pretrained('./model_test_1')
# tokenizer.save_pretrained('./model_test_1')

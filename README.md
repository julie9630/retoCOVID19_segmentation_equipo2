# Segmentación de Lesiones Pulmonares COVID-19 con U-Net

Proyecto (notebook) para segmentar lesiones en cortes de tomografía (CT) relacionadas con COVID-19 usando una arquitectura U-Net en PyTorch. Se emplea un dataset que contiene:

- `images_medseg.npy`: 100 imágenes (slices) de 512x512 (escala de grises, almacenadas como matrices con posible canal extra)
- `masks_medseg.npy`: 100 máscaras one-hot con 4 canales (clases)
- `test_images_medseg.npy`: 10 imágenes para generación de predicciones y envío (Kaggle style)

## Equipo

- Imanol Muñiz Ramirez A01701713
- Julieta Itzel Pichardo Meza
- Paul Park
- Rodrigo Antonio Benítez De La Portilla A01771433
- Carlos Iván Fonseca Mondragón A01771689

## Clases (canales de la máscara)

| Canal | Clase          | Color visualización  |
| ----- | -------------- | -------------------- |
| 0     | Ground glass   | Rojo (R)             |
| 1     | Consolidations | Verde (G)            |
| 2     | Lungs other    | Azul (B)             |
| 3     | Background     | Negro / transparente |

Para la exportación final a Kaggle solo se usan Ground glass (canal 0) y Consolidations (canal 1) en formato binario multi-canal (2 canales).

## Flujo del Notebook

1. Verificación de entorno y disponibilidad de GPU (CUDA).
2. Normalización de imágenes CT (recorte HU a [-1000, 400] y escalado a [0,1]).
3. Visualización de un slice con superposición de máscara coloreada.
4. Definición del `Dataset` personalizado (`CovidSegDataset`).
5. Implementación de U-Net con bloques `DoubleConv` (encoder–decoder + skip connections).
6. Cálculo de pesos por clase a partir de la frecuencia de píxeles (para `CrossEntropyLoss`).
7. Entrenamiento (130 épocas) con tracking de pérdida entrenamiento vs validación.
8. Inferencia sobre dataset de test con visualización de predicciones.
9. Generación de máscaras binarias (solo clases 0 y 1) para envío.
10. Creación de archivo `submission1.csv` con formato aceptado por la competencia.

## Arquitectura U-Net Implementada

- Profundidad: 4 niveles encoder + bottleneck + 4 niveles decoder
- Canales: 1 → 64 → 128 → 256 → 512 → 1024 (bottleneck) y simétrico en el decoder
- Convoluciones 3x3 (padding=1), activación ReLU
- `ConvTranspose2d` para upsampling
- Capa final 1x1: salida con 4 mapas de logits (clases)

## Función de Pérdida y Optimización

- `CrossEntropyLoss` (con pesos inversamente proporcionales a la frecuencia de cada clase)
- Optimizador: `Adam (lr=1e-4)`
- Métrica monitoreada: pérdida (train / val). (No se implementaron IoU / Dice todavía).

## Requisitos (dependencias mínimas)

Si deseas correrlo localmente:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # (ajusta según tu GPU / CUDA)
pip install numpy matplotlib pandas
```

En Kaggle normalmente ya están disponibles PyTorch, NumPy y Matplotlib.

## Estructura Recomendada (si migras desde el notebook)

```
project/
  README.md
  notebook.ipynb
  data/
    images_medseg.npy
    masks_medseg.npy
    test_images_medseg.npy
  src/
    dataset.py
    model.py
    train.py
    inference.py
```

## Ejecución Básica (Notebook)

1. Asegúrate de colocar los archivos `.npy` en la ruta correcta (en Kaggle: `/kaggle/input/covid-segmentation/`).
2. Ejecuta las celdas en orden.
3. Al finalizar el bloque de entrenamiento se mostrará la curva de pérdidas.
4. Corre las celdas de inferencia para ver las predicciones coloreadas.
5. Genera `submission1.csv` (aparecerá en el directorio de trabajo y puedes descargarlo).

## Formato de Salida (simplificado)

El código de envío actual aplana `mask_array` y genera un CSV:

```
Id,Predicted
0,0
1,1
2,0
...
```

Donde `Predicted` es cada valor binario secuencial de los dos canales concatenados (flatten). Revisa las reglas específicas de la competencia para confirmar si se requiere RLE u otro formato (este paso podría necesitar adaptación si la competencia exige codificación especial).

## Posibles Mejoras

1. Métricas adicionales: IoU (Jaccard), Dice Coefficient por clase.
2. Data augmentation (rotaciones pequeñas, flips horizontales/verticales, elastic transforms) para robustez.
3. Early stopping + checkpointing (guardar mejor modelo por `val_loss`).
4. Scheduler de LR (p.ej. `ReduceLROnPlateau`).
5. Mixed precision (torch.cuda.amp) para acelerar en GPU y reducir memoria.
6. Sustituir `CrossEntropyLoss` por combinación con `DiceLoss` o `FocalLoss` para clases desbalanceadas.
7. Post-procesamiento morfológico (apertura / cierre) para limpiar ruido en regiones pequeñas.
8. Validación cruzada (si hubiera suficientes datos) para estimar mejor la generalización.
9. Añadir logging estructurado (Weights & Biases / TensorBoard).
10. Exportar el modelo (`torch.jit.trace` o `onnx`) para despliegue.

## Ejemplo de Métrica Dice (Código de Referencia)

```python
def dice_coefficient(pred, target, num_classes=4, eps=1e-6):
    # pred: (B,C,H,W) logits -> argmax afuera
    pred_classes = torch.argmax(pred, dim=1)  # (B,H,W)
    dice_per_class = []
    for c in range(num_classes):
        pred_c = (pred_classes == c).float()
        target_c = (target == c).float()
        inter = (pred_c * target_c).sum()
        denom = pred_c.sum() + target_c.sum()
        dice = (2*inter + eps) / (denom + eps)
        dice_per_class.append(dice.item())
    return dice_per_class
```

## Consideraciones de Memoria / Rendimiento

- Batch size = 2 se eligió para evitar OOM en GPUs pequeñas.
- Si cuentas con más VRAM puedes subir a 4 u 8 y reducir épocas.
- Aumentar batch size puede requerir reajustar LR (regla lineal aproximada).

## Riesgos / Limitaciones

- Dataset pequeño (100 slices) → riesgo de sobreajuste.
- No hay separación paciente‑basada (si provinieran múltiples cortes por paciente, habría leakage si se mezclan).
- Ausencia de métricas clínicas (solo pérdida), dificulta evaluar calidad real.

## Próximos Pasos Sugeridos

- Implementar script `train.py` fuera del notebook para reproducibilidad.
- Guardar el modelo: `torch.save(model.state_dict(), 'unet_covid.pth')`.
- Añadir script `predict.py` para generar máscaras y CSV sin necesidad de ejecutar todo el notebook.

## Cita / Referencias

- Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation" (MICCAI 2015).
- Documentación PyTorch: https://pytorch.org/

## Licencia

(Si aplica, añade aquí tipo de licencia; p.ej. MIT, CC-BY, etc.)

---

Si necesitas que separe el código en módulos o agreguemos métricas adicionales, pídelo y continúo con la refactorización.

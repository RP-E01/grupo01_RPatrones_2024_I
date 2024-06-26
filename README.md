# Grupo 01 - Reconocimiento de Patrones
El reconocimiento de patrones implica la identificaci+on de regularidades, estructuras o similitudes en datos mediante el uso de algoritmos y técnicas estadísticas. Tiene como objetivo enseñar a las máquinas a entender y clasificar datos de manera automática.
Las aplicaciones van desde el análisis de imágenes y sonidos hasta el procesamiento de texto y la detección de anomalías, permitiendo que los sistemas informáticos reconozcan patrones complejos dentro de grandes volúmenes de datos y tomen decisiones basadas en esos patrones.

Como estudiantes, esperamos mejorar nuestra formación en Machine Learning, abarcando temas esenciales como aprendizaje supervisado y no supervisado, además de explorar temas más complejos como las redes neuronales.

## Presentación del equipo
###
###
### Luis Barreto

<p align="center">
  <img src="https://github.com/RP-E01/grupo01_RPatrones_2024_I/assets/86316349/e13d8c22-23fb-4d64-a83f-aa37dd1111be" alt="Luis Barreto" width="200"/>
</p>

Soy un estudiante de Ingeniería Biomédica en el noveno ciclo con un profundo interés en señales médicas e imágenes, así como en biomecánica. Siempre estoy ansioso por aprender y me motiva un objetivo principal: aprovechar la tecnología para mejorar la atención médica en Perú. Tengo una fascinación particular por la aplicación de la inteligencia artificial en el campo médico.
### Roberto Marin

<p align="center">
  <img src="https://github.com/RP-E01/grupo01_RPatrones_2024_I/assets/86316349/771314cc-c54d-42ec-b2d1-715d2f0d565b" alt="Roberto Marin" width="200"/>
</p>

Soy alumno en noveno ciclo de la carrera de Ingeniería Biomédica, quiero especializarme en el área de señales e imágenes médicas tengo interés en el uso de la inteligencia artificial para resolver problemas en el área de la salud.
### Gianfranco Feria

<p align="center">
  <img src="https://github.com/RP-E01/grupo01_RPatrones_2024_I/assets/43280063/0868f470-9304-479c-8bd7-6b3ba6571097" alt="Gianfranco Feria" width="200"/>
</p>

Estudiante de ingeniería biomédica con interés en el área de señales e imágenes, así como en el campo de la inteligencia artificial. Me considero una persona curiosa, responsable y perseverante. Mis expectativas con el proyecto son aportar al campo de la salud en nuestro país, especializándome para poder ayudar al sistema de salud, y mejorar mi formación como ingeniero.
### Arelí Sánchez

<p align="center">
  <img src="https://github.com/RP-E01/grupo01_RPatrones_2024_I/assets/135698700/b7e42a2a-ca00-49a5-a4d5-7cb08ea739e2" alt="Areli Sanchez" width="200"/>
</p>

Soy alumna del décimo ciclo de la carrera de Ingeniería Biomédica con interés en el procesamiento de señales y el desarrollo de dispositivos médicos. Me interesa idear soluciones que sean accesibles y de impacto.

## Proyecto de curso 

### Base de datos: "Neurocritical care waveform recordings in pediatric patients"

Heldt, T., Fanelli, A., Tasker, R., Vonberg, F., & LaRovere, K. (2024). Neurocritical care waveform recordings in pediatric patients (version 1.0.0). PhysioNet. https://doi.org/10.13026/5kvn-pp29.

**Publicación original**

Fanelli A, Vonberg F, LaRovere K, Walsh B, Smith E, Robinson S, Tasker RC, Heldt T. (2019). Fully automated, real-time, calibration-free, continuous noninvasive estimation of intracranial pressure in children. Journal of Neurosurgery: Pediatrics, 24(5), 509-519.

**Cita estándar**

Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online], 101(23), e215–e220.

### Descripción

La base de datos contiene grabaciones de forma de onda de pacientes pediátricos en cuidados neurocríticos, con información desidentificada y sincronizada en el tiempo. Incluye mediciones típicas obtenidas en la cabecera del paciente, como la presión arterial, la presión intracraneal y la velocidad del flujo sanguíneo cerebral. Además, proporciona datos sobre el hematocrito y la ubicación vertical (es decir, altura) de los transductores de presión arterial e intracraneal, para considerar posibles diferencias en la presión hidrostática entre ambas mediciones. La primera versión de la base de datos abarca datos de 12 pacientes con diversas patologías que requieren monitoreo invasivo de la presión intracraneal. La duración total de las grabaciones es de aproximadamente 10 horas entre todos los pacientes y estudios.

### Problemática

La monitorización de la presión intracraneal (ICP) en pacientes pediátricos es un desafío significativo debido a la naturaleza invasiva de los métodos actuales y la necesidad de adaptarse a las particularidades fisiológicas de los niños. La presión intracraneal es un parámetro crítico en el manejo de pacientes con afecciones neurológicas severas, como lesiones cerebrales traumáticas (TBI), accidentes cerebrovasculares (ACV) y hemorragias subaracnoideas (SAH). Estos parámetros son fundamentales para entender la dinámica cerebral y en la gestión de enfermedades neurológicas agudas, que son potencialmente mortales.

El ACV en pediatría, aunque menos frecuente que en adultos, constituye un desafío significativo en la neurología infantil. La incidencia anual se estima entre 2.5 y 3 casos por 100,000 niños, y aumenta con la edad. Esta patología, que incluye tanto eventos isquémicos como hemorrágicos, puede surgir en niños previamente sanos y representa una de las principales causas de muerte y discapacidad a largo plazo en este grupo etario.

A pesar de la importancia de la ICP, los métodos actuales para su monitorización en niños, especialmente en casos de ACV hemorrágico, aún requieren mejoras para adaptarse a las particularidades fisiológicas pediátricas. La detección y monitorización temprana de las alteraciones en la ICP son cruciales para la administración de tratamientos efectivos y oportunos, lo cual puede reducir significativamente la morbilidad asociada con estas condiciones.

### Objetivos

- Implementar un modelo de regresión para predecir tempranamente la ICP en situaciones críticas.
- Clasificar a los pacientes según la gravedad de su condición basándose en las características de las señales fisiológicas.
- Identificar las variables más relevantes para la predicción de la ICP mediante técnicas de selección de características.
- Validar el modelo de regresión utilizando métricas de evaluación de rendimiento como precisión y sensibilidad.
- Interpretar los resultados del modelo en el contexto clínico y discutir su aplicabilidad y limitaciones.

### Metodología

#### Preparación de Datos
- **Recolección y Acceso:** Acceder a la base de datos de PhysioNet, una fuente pública de señales fisiológicas y datos clínicos. Cargar los datos relevantes en un entorno seguro de procesamiento de datos, asegurando la conformidad con las normativas de privacidad y protección de datos.
  
- **Exploración, Limpieza y Normalización:** Realizar un análisis exploratorio inicial para identificar patrones, valores atípicos y posibles errores en los datos. Utilizar técnicas de limpieza para corregir o eliminar registros erróneos o incompletos. Normalizar las señales fisiológicas para homogeneizar la escala y facilitar su comparación y análisis.

- **Aumento de Datos:** Incrementar la diversidad y el volumen del conjunto de datos aplicando técnicas de aumento de datos a las señales fisiológicas.

#### Extracción y Selección de Características
- **Extracción:** Desarrollar algoritmos para extraer características cuantitativas de las señales fisiológicas. Implementar métodos de procesamiento de señales para extraer información relevante desde el punto de vista clínico.

- **Selección:** Emplear técnicas estadísticas y de aprendizaje automático para determinar las características más relevantes para los modelos predictivos.

#### Modelado y Validación
- **Modelos de Aprendizaje Automático:** Utilizar modelos LSTM para regresión y desarrollar modelos de clasificación para ICP lateralizada. Mejorar la precisión y estabilidad con modelos de ensamble.

- **Evaluación:** Evaluar el rendimiento de los modelos utilizando métricas específicas como precisión, sensibilidad, especificidad y el área bajo la curva ROC (AUC).

#### Aplicación y Documentación
- **Interpretación de Resultados**: Analizar e interpretar los resultados en el contexto clínico.
- **Desarrollo de Herramientas**: Crear prototipos de herramientas basadas en los modelos para asistencia clínica.
- **Documentación**: Elaborar documentación sobre los métodos, modelos, y resultados obtenidos.

#### Consideraciones Éticas
- Asegurar la protección de los datos de pacientes y seguir directrices éticas en la investigación.


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

La detección temprana de crisis epilépticas en pacientes pediátricos con epilepsia es un problema significativo debido a varias razones. En primer lugar, las crisis epilépticas son una causa frecuente de consulta en la emergencia y en la atención ambulatoria. Se estima que aproximadamente el 10% de la población general tendrá una crisis epiléptica en su vida y la mitad de ellas ocurrirá en la infancia y adolescencia. 

Además, la ocurrencia de una crisis no implica necesariamente el diagnóstico de epilepsia, y el tratamiento subsecuente. Por lo tanto, es crucial poder predecir y detectar estas crisis para administrar el tratamiento adecuado y evitar posibles complicaciones. Por otro lado, no todo paciente con primera crisis debe ser dejado en observación sin recibir el manejo apropiado. Esta decisión está en función del riesgo de recurrencia de crisis. Por lo tanto, un modelo de machine learning que pueda predecir estas crisis basado en los datos de formas de onda del cerebro puede ser de gran ayuda para los médicos y cuidadores.

Finalmente, las crisis epilépticas pueden tener un impacto significativo en la calidad de vida del paciente. Por lo tanto, cualquier herramienta que pueda ayudar a predecir y, por lo tanto, a manejar mejor estas crisis, puede tener un impacto positivo en la vida del paciente. Sin embargo, es importante tener en cuenta que cualquier modelo desarrollado con estos datos necesitaría ser validado en estudios clínicos antes de poder ser utilizado en la práctica.

### Objetivos

- Predicción de eventos adversos: Utilizar las señales fisiológicas para predecir la ocurrencia de eventos adversos como el aumento de la presión intracraneal o ladisminución del flujo sanguíneo cerebral.
- Clasificación de pacientes: Clasificar a los pacientes en grupos según la gravedad de su condición o la respuesta al tratamiento, basándose en las característicasde las señales fisiológicas.
- Detección temprana de deterioro: Identificar patrones en las señales fisiológicas que indiquen un deterioro temprano en la condición del paciente, permitiendo intervenciones tempranas.

### Metodología

#### Preparación de Datos
- **Recolección y Acceso**: Obtener acceso y cargar la base de datos de PhysioNet.
- **Limpieza y Normalización**: Aplicar técnicas de limpieza para corregir o eliminar datos erróneos o incompletos. Normalizar las señales para su análisis.

#### Extracción y Selección de Características
- **Extracción**: Diseñar métodos para extraer características relevantes de las señales fisiológicas.
- **Selección**: Utilizar técnicas para identificar las características más significativas para los objetivos del estudio.

#### Modelado y Validación
- **Modelos de Machine Learning**: Seleccionar y entrenar modelos adecuados, utilizando una división de datos en entrenamiento y prueba, y aplicar validación cruzada.
- **Evaluación**: Usar métricas como precisión, sensibilidad, y especificidad para evaluar el rendimiento de los modelos.

#### Aplicación y Documentación
- **Interpretación de Resultados**: Analizar e interpretar los resultados en el contexto clínico.
- **Desarrollo de Herramientas**: Crear prototipos de herramientas basadas en los modelos para asistencia clínica.
- **Documentación**: Elaborar documentación sobre los métodos, modelos, y resultados obtenidos.

#### Consideraciones Éticas
- Asegurar la protección de los datos de pacientes y seguir directrices éticas en la investigación.


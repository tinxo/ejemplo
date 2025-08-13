Este proyecto implementa una segunda iteración del enfoque CRISP-DM, incorporando prácticas de DataOps y MLOps:

- Tests automáticos de calidad de datos con Pandera
- Versionado de datos con DVC
- Experimentación y tracking con MLflow
- Despliegue de predicciones vía API y UI

----

Estructura:

- data: contiene los datos del proyecto
  - raw: versiones originales de los datos sin modificar
  - processed: versiones modificadas en el marco del proyecto
- notebooks: espacio para experimentación (generalmente con jupyter notebooks, de ahí el nombre)
- src: directorio para los scripts a generar del proyecto a partir del contenido de las notebooks
- tests: ubicación de los tests a ejecutar para el proyecto (archivos test_###.py)
- models: espacio para almacenar los modelos que puedan ser generados por el proyecto
- app: contiene la versión de la aplicación que permite usar el modelo

----


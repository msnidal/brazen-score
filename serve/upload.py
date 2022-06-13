from google.cloud import aiplatform

VERSION = 1
MODEL_NAME = "brazen-score"
model_display_name = f"{MODEL_NAME}-v{VERSION}"
model_description = "Brazen Score"

CUSTOM_PREDICTOR_IMAGE_URI = "northamerica-northeast1-docker.pkg.dev/brazen-score/brazen-score/brazen-score"

health_route = "/ping"
predict_route = f"/predictions/{MODEL_NAME}"
serving_container_ports = [8080]

model = aiplatform.Model.upload(
    display_name=model_display_name,
    description=model_description,
    serving_container_image_uri=CUSTOM_PREDICTOR_IMAGE_URI,
    serving_container_predict_route=predict_route,
    serving_container_health_route=health_route,
    serving_container_ports=serving_container_ports,
)

model.wait()

print(model.display_name)
print(model.resource_name)

Как запустить?
Запуск проекта (Docker)

1) `docker pull fanest1i/linguo-image-ml-service:latest`
2) `docker run -p 8000:8000 --env-file .env -v linguo-ml-models:/models --name linguo-ml fanest1i/linguo-image-ml-service:latest`

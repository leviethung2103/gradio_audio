
## Build Docker Image
```
docker build . -t gradio-app
```

## Run 
```
docker run --rm -it -p 8080:8080 gradio_app:latest
```

# Hot reload mode
```bash
gradio main.py
```
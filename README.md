# VK-бот на базе LLM

Для создания docker-образа - запустить команду

```bash
sudo sh build_image.sh
```

Для запуска образа - выполнить команду

```bash
sudo docker run --rm -it --gpus all --name vk -e VK_API_KEY=xxx yurkoff/vk-llm:0.1-gpu
```

, где `xxx` - API-ключ для группы в VK.
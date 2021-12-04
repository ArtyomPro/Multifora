# Готовая сборка

### Requirements
- doker. [Инструкция по установке](https://bit.ly/3EotkbD)
- docker-compose. [Инструкция по установке](https://bit.ly/3djU8hd)

### Инструкция по развертыванию:
- 1) Скопируйте данную папку к себе на сервер при помощи команды ```git clone```
- 2) Выполните команду в терминале ``` mkdir ./DBMongo``` для создания volume для mongodb. (p.s. можно создать папку в другом месте или использовать готовую бд, однако, тогда замените параметр ```volumes``` в файле ```docker-compose.yml```)
- 3) Выполните команду в терминале ```docker-compose up -d```
- 4) Сборка готова!

## Images:
- 1) Image контейнера multifora-flask, который принимает запросы клиента и передает их multifora-worker для выполнения таски, находится по [ссылке](https://hub.docker.com/repository/docker/fastrus1804/multifora-flask).
- 2) Image контейнера multifora-worker, который выполняет таски из multifora-flask. (Например, добавление модели, обучения модели, и т.д.), находится по [ссылке](https://hub.docker.com/repository/docker/fastrus1804/multifora-worker).

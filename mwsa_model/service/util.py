from enum import Enum

import yaml


class SupportedLanguages(Enum):
    Basque = 'eu'
    Bulgarian = 'bg'
    Danish = 'da'
    Dutch = 'nl'
    English = 'en'
    Estonian = 'et'
    Hungarian = 'hu'
    German = 'de'
    Irish = 'ga'
    Italian = 'it'
    Portuguese = 'pt'
    Russian = 'ru'
    Serbian = 'sr'
    Slovene = 'sl'


def load_dvc_params(lang) -> dict:
    config_file = lang + '_params.yaml'

    with open(config_file, 'r') as fd:
        params = yaml.safe_load(fd)

        return params

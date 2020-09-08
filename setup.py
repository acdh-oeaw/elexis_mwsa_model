import setuptools

setuptools.setup(
      name='mwsa_model',
      version=1.0,
      author='Seung-bin Yim, Lenka Bajcetic',
      author_email='seung-bin.yim@oeaw.ac.at',
      description="Elexis Monolingual Word sense alignment models",
      url='https://github.com/acdh-oeaw/elexis_mwsa_model',
      packages=['mwsa_model.transformers','mwsa_model.service'],
      install_requires=[
            'spacy-wordnet==0.0.4',
            'en-core-web-md@https://github.com/explosion/spacy-models/releases/download/en_core_web_md-2.3.1/en_core_web_md-2.3.1.tar.gz',
            'de_core_news_md@https://github.com/explosion/spacy-models/releases/download/de_core_news_md-2.3.0/de_core_news_md-2.3.0.tar.gz'
      ]
)

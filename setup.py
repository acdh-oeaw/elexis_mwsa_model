import setuptools

setuptools.setup(
      name='mwsa_model',
      version=1.0,
      author='Seung-bin Yim, Lenka Bajcetic',
      author_email='seung-bin.yim@oeaw.ac.at',
      description="Elexis Monolingual Word sense alignment models",
      url='https://github.com/acdh-oeaw/elexis_mwsa_model',
      packages=['mwsa_model.transformers','mwsa_model.service']
)

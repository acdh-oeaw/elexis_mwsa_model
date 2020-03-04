
class ClassifierConfig:
    def __init__(self, language_model, language, folder, balancing_strategy='oversampling', testset_ratio=0.0,
                 with_testset=False):
        self.language_model = language_model
        self.language = language
        self.folder = folder
        self.with_testset = with_testset
        self.testset_ratio = testset_ratio
        self.balancing_strategy = balancing_strategy


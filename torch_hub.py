from torch import hub

class TorchHub:
    def __init__(self, repo_name: str, entry_point: str):      

        assert repo_name, "repo_name is required"
        assert entry_point, "entry_point is required"

        self.repo_name = repo_name
        self.entry_point = entry_point        

    def get_model(self):
        model = hub.load(self.repo_name, self.entry_point)
        print(model)
        return model
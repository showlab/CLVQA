# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import registry
from mmf.datasets.builders.clvqa.dataset import CLVQADataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("clvqa")
class CLVQABuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="clvqa", dataset_class=CLVQADataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/clvqa/defaults.yaml"

    # TODO: Deprecate this method and move configuration updates directly to processors
    def update_registry_for_model(self, config):
        if hasattr(self.dataset, "text_processor"):
            registry.register(
                self.dataset_name + "_text_vocab_size",
                self.dataset.text_processor.get_vocab_size(),
            )
            registry.register(
                f"{self.dataset_name}_text_processor", self.dataset.text_processor
            )
        if hasattr(self.dataset, "answer_processor"):
            registry.register(
                self.dataset_name + "_num_final_outputs",
                self.dataset.answer_processor.get_vocab_size(),
            )
            registry.register(
                f"{self.dataset_name}_answer_processor", self.dataset.answer_processor
            )

    # overide this function in MMFDatasetBuilder
    def load(self, config, dataset_type, *args, **kwargs):
        self.config = config
        split_dataset_from_train = self.config.get("split_train", False)
        if split_dataset_from_train:
            config = self._modify_dataset_config_for_split(config)
        
        annotations = self._read_annotations(config, dataset_type)
        if annotations is None:
            return None

        dataset_class = self.dataset_class
        dataset = dataset_class(config, dataset_type, 0)
        
        if split_dataset_from_train:
            dataset = self._split_dataset_from_train(dataset, dataset_type)

        self.dataset = dataset
        return self.dataset
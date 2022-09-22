import random
import logging
import torch.utils.data
from pathlib import Path
from trackron.structures import TensorDict
from fvcore.common.registry import Registry
from trackron.config import configurable
from trackron.config import configurable
from .trainsets import get_trainsets
from ..processing import build_processing_class
from .build import DATASET_REGISTRY, no_processing


@DATASET_REGISTRY.register()
class SequenceDataset(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. Each training sample is a
    tuple consisting of i) a set of template frames, used to learn the DiMP classification model and obtain the
    modulation vector for IoU-Net, and ii) a set of search frames on which target classification loss for the predicted
    DiMP model, and the IoU prediction loss for the IoU-Net is calculated.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'template frames' and
    'search frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    @configurable
    def __init__(self,
                 datasets,
                 p_datasets,
                 samples_per_epoch,
                 max_gap,
                 num_search_frames,
                 processing=no_processing,
                 frame_sample_mode='casual'):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the template frames and the search frames.
            num_search_frames - Number of search frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'casual' or 'interval'. If 'casual', then the search frames are sampled in a casually,
                                otherwise randomly within the interval.
        """
        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_search_frames = num_search_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode

    @classmethod
    def from_config(cls, cfg, training=False):
        if training:
            datasets = get_trainsets(cfg.TRAIN.DATASET_NAMES, cfg.ROOT)
            p_datasets = cfg.TRAIN.DATASETS_RATIO
            frames = cfg.SEARCH.FRAMES
        else:
            datasets = get_trainsets(cfg.VAL.DATASET_NAMES, cfg.ROOT)
            p_datasets = cfg.VAL.DATASETS_RATIO
            frames = cfg.VAL.FRAMES
        # TRACED: 数据处理的方式！
        processing_class = build_processing_class(cfg, training)
        return {
            "datasets": datasets,
            "p_datasets": p_datasets,
            "samples_per_epoch": cfg.TRAIN.SAMPLE_PER_EPOCH,
            "max_gap": cfg.MAX_SAMPLE_INTERVAL,
            "num_search_frames": frames,
            "processing": processing_class,
            "frame_sample_mode": cfg.SAMPLE_MODE,
        }

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self,
                            visible,
                            num_ids=1,
                            min_id=None,
                            max_id=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return list(sorted(random.choices(valid_ids, k=num_ids)))

    def __getitem__(self, index):
        """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks
        """

        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]
        is_video_dataset = dataset.is_video_sequence()
        retry_count = 0

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while True:
            while not enough_visible_frames:
                # Sample a sequence
                seq_id = random.randint(0, dataset.get_num_sequences() - 1)

                # Sample frames
                seq_info_dict = dataset.get_sequence_info(seq_id)
                visible = seq_info_dict['visible']

                enough_visible_frames = visible.type(torch.int64).sum().item(
                ) > 2 * self.num_search_frames and len(visible) >= 20

                enough_visible_frames = enough_visible_frames or not is_video_dataset

            if is_video_dataset:
                search_frame_ids = None
                gap_increase = 0

                if self.frame_sample_mode == 'interval':
                    # Sample frame numbers within interval defined by the first frame
                    while search_frame_ids is None:
                        base_frame_id = self._sample_visible_ids(visible,
                                                                 num_ids=1)
                        search_frame_ids = self._sample_visible_ids(
                            visible,
                            num_ids=self.num_search_frames,
                            min_id=base_frame_id[0] - self.max_gap -
                            gap_increase,
                            max_id=base_frame_id[0] + self.max_gap +
                            gap_increase)
                        gap_increase += 5  # Increase gap until a frame is found
                elif self.frame_sample_mode == 'casual':
                    while search_frame_ids is None:
                        search_frame_ids = self._sample_visible_ids(
                            visible, num_ids=self.num_search_frames)
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                search_frame_ids = [1] * self.num_search_frames

            search_frames, search_anno, meta_obj_search = dataset.get_frames(
                seq_id, search_frame_ids, seq_info_dict)

            # TRACED: coco 数据集格式 480*640
            H, W, _ = search_frames[0].shape
            search_masks = search_anno['mask'] if 'mask' in search_anno else [
                torch.zeros((H, W))
            ] * self.num_search_frames

            data = TensorDict({
                'search_images':
                search_frames,
                'search_boxes':
                search_anno['bbox'],
                'search_masks':
                search_masks,
                'dataset':
                dataset.get_name(),
                'search_class':
                meta_obj_search.get('object_class_name')
            })
            try:
                # BUG: 无法成功加载数据！原因：将图像 crop 操作去掉！即 utt.yaml 中 DATASET.SEARCH.SIZE=None 默认就不进行裁剪，只进行 padding 操作
                # data['template_images'] 去了哪里？如果只有 search_images 如何进行训练？
                return self.processing(data)
            except Exception as e:
                retry_count += 1
                enough_visible_frames = False
                logger = logging.getLogger(__name__)
                logger.warning("failed retry data {}".format(retry_count))
                logger.warning(e)

            # return self.processing(data)

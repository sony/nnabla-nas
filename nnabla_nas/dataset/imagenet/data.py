from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types


_pixel_mean = [255 * x for x in (0.485, 0.456, 0.406)]
_pixel_std = [255 * x for x in (0.229, 0.224, 0.225)]


class TrainPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, shard_id, image_dir, file_list,
                 nvjpeg_padding, prefetch_queue=3, seed=1, num_shards=1,
                 channel_last=True, dtype="half"):
        super(TrainPipeline, self).__init__(
            batch_size, num_threads, shard_id, seed=seed,
            prefetch_queue_depth=prefetch_queue)
        self.input = ops.FileReader(file_root=image_dir, file_list=file_list,
                                    random_shuffle=True, num_shards=num_shards,
                                    shard_id=shard_id)
        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB,
                                        device_memory_padding=nvjpeg_padding,
                                        host_memory_padding=nvjpeg_padding)

        self.rrc = ops.RandomResizedCrop(device="gpu", size=(224, 224))
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT16
                                            if dtype == "half"
                                            else types.FLOAT,
                                            output_layout=types.NHWC
                                            if channel_last else types.NCHW,
                                            crop=(224, 224),
                                            image_type=types.RGB,
                                            mean=_pixel_mean,
                                            std=_pixel_std,
                                            pad_output=False)
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.rrc(images)
        images = self.cmnp(images, mirror=self.coin())
        return images, labels.gpu()


class ValPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, shard_id, image_dir, file_list,
                 nvjpeg_padding, seed=1, num_shards=1, channel_last=True,
                 dtype='half'):
        super(ValPipeline, self).__init__(
            batch_size, num_threads, shard_id, seed=seed)
        self.input = ops.FileReader(file_root=image_dir, file_list=file_list,
                                    random_shuffle=False,
                                    num_shards=num_shards, shard_id=shard_id)
        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB,
                                        device_memory_padding=nvjpeg_padding,
                                        host_memory_padding=nvjpeg_padding)
        self.res = ops.Resize(device="gpu", resize_shorter=256)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT16 if
                                            dtype == "half" else types.FLOAT,
                                            output_layout=types.NHWC
                                            if channel_last else types.NCHW,
                                            crop=(224, 224),
                                            image_type=types.RGB,
                                            mean=_pixel_mean,
                                            std=_pixel_std,
                                            pad_output=False)

    def define_graph(self):
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.res(images)
        images = self.cmnp(images)
        return images, labels.gpu()

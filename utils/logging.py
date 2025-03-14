import wandb
import torch
from torch.utils.tensorboard import SummaryWriter

from .build_opt import Opt

class Logging:
    def __init__(self, opt: Opt):
        self.__opt = opt
        self.__args = opt.args
        self.__epoch = 0  # Luôn đảm bảo step tăng dần
        self.__run = None
        self.__writer = None

        # Khởi tạo WandB nếu được bật
        if self.__args.wandb:
            if self.__args.mode == "training":
                self.__args.run_name = f"{self.__args.ds}_{self.__args.model_type}_unet{self.__args.unet_number}_condscale{self.__args.cond_scale_validation}"
            elif self.__args.mode == "testing":
                self.__args.run_name = f"{self.__args.ds}_{self.__args.model_type}_unet{self.__args.unet_number}_condscale{self.__args.cond_scale_testing}"

            self.__run = wandb.init(
                project=self.__args.wandb_prj,
                name=self.__args.run_name,
                config=vars(self.__args),
                resume="allow"
            )
            self.__run.define_metric("*", step_metric="global_step")

        # Khởi tạo TensorBoard nếu được bật
        if self.__args.log:
            self.__writer = SummaryWriter(self.__args.exp_dir)
            self.__writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.__args).items()])),
            )

    def __call__(self, key, value, step=None, metadata=None):
        if self.__run is None and self.__writer is None:
            return  # Không có logging nào được bật, bỏ qua

        if step is not None:
            self.__epoch = max(self.__epoch, step)  # Cập nhật step để không bị lỗi

        log_data = {key: value}
        if metadata:
            log_data.update(metadata)

        if self.__run:  # Log vào WandB nếu bật
            log_data["global_step"] = self.__epoch
            self.__run.log(log_data)

        if self.__writer:  # Log vào TensorBoard nếu bật
            self.__writer.add_scalar(key, value, global_step=self.__epoch)

    def log_iteration(self, iteration):
        """Cập nhật số iteration chính xác"""
        self.__epoch = max(self.__epoch, iteration)

    def watch(self, model):
        if self.__run:
            self.__run.watch(models=model, log="all", log_freq=self.__args.num_train_batch, log_graph=True)

    def log_images(self, images_dict, iteration):
        """Log hình ảnh với step đúng"""
        if self.__run is None:
            return  # Không có WandB, bỏ qua

        iteration = max(self.__epoch, iteration)
        log_dict = {"global_step": iteration}

        for prompt, data in images_dict.items():
            generated_image = data.get("Generated Image", None)
            if generated_image is not None:
                log_dict[f"Generated/{prompt}"] = wandb.Image(generated_image, caption=f"Step {iteration}")

        self.__run.log(log_dict)

    def close(self):
        if self.__run:
            self.__run.finish()

        if self.__writer:
            self.__writer.close()

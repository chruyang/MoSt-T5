from transformers import T5Config

class MoStT5Config(T5Config):
    """
    MoSt-T5 模型配置类
    继承自 T5Config，增加了 3D E3FP 相关的特定参数。
    """
    model_type = "most-t5"

    def __init__(self,
                 e3fp_vocab_size=4096,
                 e3fp_num_levels=4, # Level 3 + 1 (Base)
                 fusion_type="residual", # 融合方式: residual 或 gate
                 **kwargs):
        """
        :param e3fp_vocab_size: E3FP 指纹的 Bit 数 (不含 Padding)
        :param e3fp_num_levels: E3FP 的层级数 (Level 0,1,2,3)
        :param fusion_type: 融合层类型
        """
        super().__init__(**kwargs)
        self.e3fp_vocab_size = e3fp_vocab_size
        self.e3fp_num_levels = e3fp_num_levels
        self.fusion_type = fusion_type
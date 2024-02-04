import torch


from torchao.quantization.utils import (
    apply_logging_hook,
    compute_error,
    compute_error as SQNR,
    fqn_to_op_to_shape_to_count,
    LoggingTensorMode,
)

from torchao.quantization.subclass import Int8DynamicallyQuantizedLinearWeight, Int8WeightOnlyQuantizedLinearWeight


import logging


# set logger format with filename and line number
logging.basicConfig(
    format="%(filename)s:%(lineno)d - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class TestCase:

    def assertGreater(self, a, b, msg):
        assert a > b, msg

    def _test_lin_weight_subclass_impl(
        self,
        test_subclass_from_float,
        min_sqnr=35,
        test_dtype=torch.bfloat16,
        test_shape=(16, 32, 64),
    ):

        # x: [bs, in_feats]
        # weight: [out_feats, in_feats]
        # output: [bs, out_feats]

        bs, in_feats, out_feats = test_shape
        x = torch.randn(bs, in_feats, device="cuda", dtype=test_dtype)
        lin = torch.nn.Linear(in_feats, out_feats, device="cuda").to(test_dtype)
        ref_f = lin(x)

        lin.weight = torch.nn.Parameter(
            test_subclass_from_float(lin.weight), requires_grad=False
        )
        test = lin(x)
        self.assertGreater(
            SQNR(ref_f, test),
            min_sqnr,
            f"{lin.weight.__class__.__name__} failed, no compile, dtype={test_dtype}, (bs, in_feats, out_feats)={test_shape}"
        )
        # Test for compile
        # lin_comp = torch.compile(
        #     lin,
        #     #  mode='max-autotune'
        # )
        # test_comp = lin_comp(x)
        # self.assertGreater(
        #     SQNR(ref_f, test_comp),
        #     min_sqnr,
        #     f"{lin.weight.__class__.__name__} failed at compile with dtype={test_dtype}, (bs, in_feats, out_feats)={test_shape}"
        # )


test_cases = TestCase()
test_dtype = torch.float32
test_cases._test_lin_weight_subclass_impl(
    Int8WeightOnlyQuantizedLinearWeight.from_float,
    min_sqnr=35,
    test_dtype=torch.float32,
    test_shape=(16, 32, 64),
)





from typing import Union, Tuple





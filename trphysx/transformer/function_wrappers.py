from collections import OrderedDict

class AutoGenerationWrapper:

    @classmethod
    def cylinder_wrapper(cls, func):
        '''
        Cylinder function wrapper for the forward pass of the transformer model
        during time-series generation
        '''
        def run(*args, **kwargs):
            output = func(*args, **kwargs)
            return output
            # Force viscosity of the generated time-series to match the input
            # out_embeddings = output[0]
            # out_embeddings[:, :, -1] = kwargs['inputs_embeds'][:, 0, -1].unsqueeze(-1).repeat(1, out_embeddings.size(1))
            # output = list(output)
            # output[0] = out_embeddings
            # return tuple(output)
        return run

    @classmethod
    def lorenz_wrapper(cls, func):
        '''
        Lorenz function wrapper for the forward pass of the transformer model
        during time-series generation
        '''
        def run(*args, **kwargs):
            return  func(*args, **kwargs)
        return run

    @classmethod
    def default_wrapper(cls, func):
        '''
        Default function wrapper for the forward pass of the transformer model
        '''
        def run(*args, **kwargs):
            return  func(*args, **kwargs)
        return run

    @classmethod
    def forward_wrapper(cls, model_name, func):
        WRAPPER_MAPPING = OrderedDict(
            [
                ("lorenz", cls.lorenz_wrapper),
                ("cylinder", cls.cylinder_wrapper),
            ]
        )
        # First check if the model name is a pre-defined config
        if (model_name in WRAPPER_MAPPING.keys()):
            wrapper = WRAPPER_MAPPING[model_name]
            # Init config class
            func_wrap = wrapper(func)
        else:
            func_wrap = cls.default_wrapper(func)

        return func_wrap
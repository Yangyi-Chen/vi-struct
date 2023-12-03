def add_space_to_tokenizer(tokenizer):
    # https://github.com/salesforce/CodeGen/blob/91c58d5fe903b9a662fbfb5fda164417db44d048/jaxformer/hf/sample.py#L74
    def include_whitespace(t, n_min=2, n_max=20, as_special_tokens=False):
        t.add_tokens([' ' * n for n in reversed(range(n_min, n_max))], special_tokens=as_special_tokens)
        return t

    def include_tabs(t, n_min=2, n_max=20, as_special_tokens=False):
        t.add_tokens(['\t' * n for n in reversed(range(n_min, n_max))], special_tokens=as_special_tokens)
        return t
        
    tokenizer = include_whitespace(t=tokenizer, n_min=2, n_max=32, as_special_tokens=False)
    tokenizer = include_tabs(t=tokenizer, n_min=2, n_max=10, as_special_tokens=False)
    return tokenizer

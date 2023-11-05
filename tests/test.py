import torch
import torch.nn.functional as F

from transformers import AutoTokenizer

from tokens_in_common.data import OptionStringBuildMode, multitext_from_option_strings
from tokens_in_common.modeling.llama import LlamaModel
from tokens_in_common.tokenization import tokenize_multitext


def test_standard_full_equivalency():
    MODEL_KEY = 'meta-llama/Llama-2-7b-hf'

    # TEST DATA
    test_multitexts = {}

    # easy test case
    TEST_SAMPLE1 = ['The sentence "Four children are playing in some water." is ', ('true.', 'false.'),
                    'The sentence "The children are wet." is ', ('true.', 'false.')]
    test_multitexts['1_full'] = multitext_from_option_strings(OptionStringBuildMode.FULL, TEST_SAMPLE1)
    test_multitexts['1_standard'] = multitext_from_option_strings(OptionStringBuildMode.STANDARD, TEST_SAMPLE1)
    test_multitexts['1_frugal'] = multitext_from_option_strings(OptionStringBuildMode.FRUGAL, TEST_SAMPLE1)

    # TODO:  harder test case
    #

    TEST_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_KEY)
    TEST_MODEL = LlamaModel.from_pretrained(MODEL_KEY)

    output_dict = {}
    for key, test_multitext in test_multitexts.items():
        # tokenize the multitext
        tok_multitext = tokenize_multitext(test_multitext, lambda x: TEST_TOKENIZER(x).encodings[0])
        max_position = max(v.position for v in tok_multitext.vertices)

        # forward model
        tokens, positions, attention_mask, token_vertex_elements = tok_multitext.prepare_inputs()

        t_tokens = torch.LongTensor(tokens).unsqueeze(dim=0)
        t_positions = torch.LongTensor(positions).unsqueeze(dim=0)
        t_attention_mask = torch.LongTensor(attention_mask).unsqueeze(dim=0).unsqueeze(dim=0)

        outputs = TEST_MODEL(
            input_ids=t_tokens, attention_mask=t_attention_mask, position_ids=t_positions,
            output_hidden_states=True
        )

        for i, (_, pos, anc_hash) in enumerate(token_vertex_elements):
            if pos == max_position:
                attn = t_attention_mask[0, :, i].bool()
                toks = t_tokens[0, i]
                tpos = t_positions[0, i]
                attended_tokens = t_tokens[attn]
                attended_positions = t_positions[attn]
                state = outputs.hidden_states[-1][0, i]
                if key not in output_dict:
                    output_dict[key] = {}
                if anc_hash not in output_dict[key]:
                    output_dict[key][anc_hash] = []
                output_dict[key][anc_hash].append((toks, tpos, attn, attended_tokens, attended_positions, state))

    # COMPARE BUILD_MODE OUTPUTS
    a_dict = output_dict['1_full']
    b_dict = output_dict['1_standard']
    for anc_hash in a_dict.keys():
        for i, (a, b) in enumerate(zip(a_dict[anc_hash], b_dict[anc_hash])):
            cos_sim = F.cosine_similarity(a[-1], b[-1], dim=0)
            print(anc_hash, i, cos_sim)
            assert torch.allclose(a[-1], b[-1], atol=1e-5)

    c_dict = output_dict['1_frugal']
    for anc_hash in a_dict.keys():
        for i, (a, c) in enumerate(zip(a_dict[anc_hash], c_dict[anc_hash])):
            cos_sim = F.cosine_similarity(a[-1], c[-1], dim=0)
            print(anc_hash, i, cos_sim)


from transformers import AutoTokenizer

from tokens_in_common.data import OptionStringBuildMode, multitext_from_option_strings
from tokens_in_common.tokenization import tokenize_multitext

DIRECTORY = 'images/'

test_multitexts = {}

# easy test case
TEST_SAMPLE = ['The sentence "Four children are playing in some water." is ', ('true.', 'false.'),
               'The sentence "The children are wet." is ', ('true.', 'false.')]
test_multitexts['1_full'] = multitext_from_option_strings(OptionStringBuildMode.FULL, TEST_SAMPLE)
test_multitexts['1_standard'] = multitext_from_option_strings(OptionStringBuildMode.STANDARD, TEST_SAMPLE)
test_multitexts['1_frugal'] = multitext_from_option_strings(OptionStringBuildMode.FRUGAL, TEST_SAMPLE)

TEST_SAMPLE_DISCOURSE = [
    'The sentence "Four children are playing in some water." is true.',
    ('Therefore, t', 'T'), 'he sentence "The children are wet." is ', ('true.', 'false.')
]
test_multitexts['discourse_standard'] = multitext_from_option_strings(
    OptionStringBuildMode.STANDARD, TEST_SAMPLE_DISCOURSE
)

MODEL_KEY = 'meta-llama/Llama-2-7b-hf'
TEST_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_KEY)
RENDER_KWARGS = dict(format='png', directory=DIRECTORY, cleanup=True)


def render(key, test_multitext):
    # render the multitext
    test_multitext.render_with_graphviz(f'{key}_str_multitext', **RENDER_KWARGS)

    # tokenize the multitext
    tok_multitext = tokenize_multitext(test_multitext, lambda x: TEST_TOKENIZER(x).encodings[0])

    # render the tokenized multitext
    tok_multitext.render_with_graphviz(
        f'{key}_token_multitext',
        label_fn=lambda v: str(TEST_TOKENIZER.convert_ids_to_tokens(v)),
        **RENDER_KWARGS
    )


for key in test_multitexts.keys():
    render(key, test_multitexts[key])


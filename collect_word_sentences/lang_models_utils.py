import numpy
import os
import random
import re
import torch

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

class ContextualizedModelCard:
    def __init__(self, args, causal=False):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        self.causal = causal
        self.model_name, self.to_cuda = self.read_names(args)
        self.cuda_device = 'cuda:{}'.format(0)
        self.model = self.load_model()
        self.required_shape, self.max_len, self.n_layers = self.read_details()
        self.tokenizer = self.load_tokenizer()

    def read_names(self, args):
        to_cuda = False
        if 'minerva' in args.model:
            if args.model == 'minervapt-3b':
                model_name = "sapienzanlp/Minerva-3B-base-v1.0"
            if args.model == 'minervapt-1b':
                model_name = "sapienzanlp/Minerva-1B-base-v1.0"
            if args.model == 'minervapt-350m':
                model_name = "sapienzanlp/Minerva-350M-base-v1.0"
        to_cuda = True
        ### gpt2
        if 'gpt2' in args.model:
            if args.lang == 'de':
                if 'small' in args.model:
                    model_name = "benjamin/gerpt2"
                else:
                    model_name = "benjamin/gerpt2-large"
            elif args.lang == 'it':
                #model_name = "GroNLP/gpt2-medium-italian-embeddings"
                model_name = 'LorenzoDeMattei/GePpeTto'
            to_cuda = True
        ### llama 1b
        if args.model == 'llama-1b':
            model_name = "meta-llama/Llama-3.2-1B"
            to_cuda = True
        ### llama 3b
        if args.model == 'llama-3b':
            model_name = "meta-llama/Llama-3.2-3B"
            to_cuda = True
        ### XLM-roberta
        if 'large' in args.model:
            model_name = args.model
            to_cuda =True
        if 'xxl' in args.model:
            model_name = 'facebook/{}'.format(args.model)
            to_cuda = True
        if 'rta-xl' in args.model:
            model_name = 'facebook/{}'.format(args.model)
            to_cuda = True
        ### XGLM
        if args.model == 'xglm-564m':
            model_name = "facebook/xglm-564M"
            to_cuda = True
        if args.model == 'xglm-1.7b':
            model_name = "facebook/xglm-1.7B"
            if torch.cuda.get_device_properties(int(0)).total_memory/1e6 > 20000:
                to_cuda = True
        if args.model == 'xglm-2.9b':
            model_name = "facebook/xglm-2.9B"
            if torch.cuda.get_device_properties(int(0)).total_memory/1e6 > 20000:
                to_cuda = True
        if args.model == 'xglm-4.5b':
            model_name = "facebook/xglm-4.5B"
            if torch.cuda.get_device_properties(int(0)).total_memory/1e6 > 20000:
                to_cuda = True
        if args.model == 'xglm-7.5b':
            model_name = "facebook/xglm-7.5B"
        return model_name, to_cuda

    def load_model(self):
        cache = os.path.join('/', 'data', 'tu_bruera', 'hf_models')
        os.makedirs(cache, exist_ok=True)
        if not self.causal:
            model = AutoModel.from_pretrained(
                                          self.model_name, 
                                          cache_dir=cache,
                                          )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                                          self.model_name, 
                                          cache_dir=cache,
                                          )
        if self.to_cuda:
            model.to(self.cuda_device)
        return model

    def read_details(self):
        if 'xglm' in self.model_name:
            required_shape = self.model.config.d_model
            max_len = self.model.config.max_position_embeddings
            n_layers = self.model.config.num_layers
        elif 'xlm' in self.model_name or 'llama' in self.model_name or 'pt' in self.model_name or 'Ge' in self.model_name or 'iner' in self.model_name:
            required_shape = self.model.config.hidden_size
            max_len = self.model.config.max_position_embeddings
            n_layers = self.model.config.num_hidden_layers
        print('Dimensionality: {}'.format(required_shape))
        print('Number of layers: {}'.format(n_layers))

        return required_shape, max_len, n_layers

    def load_tokenizer(self):
        cache = os.path.join('/', 'data', 'tu_bruera', 'hf_tokenizers')
        os.makedirs(cache, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(
                                                  self.model_name, 
                                                  cache_dir=cache,
                                                  sep_token='[SEP]',
                                                  max_length=self.max_len,
                                                  truncation=True,
                                                  clean_up_tokenization_spaces=True,
                                                  )
        return tokenizer

def read_all_sentences(args):
    all_sentences = dict()
    sentences_folder = os.path.join('sentences', args.lang, args.corpus)

    for f in os.listdir(sentences_folder):
        if 'tsv' not in f:
            continue
        word = f.split('.')[0]
        with open(os.path.join(sentences_folder, f)) as i:
            lines = [l.strip().split('\t') for l in i.readlines()]
            ### removing cases where less than 5 words come before the actual word
            for l in lines:
                if l.index(word) < 5:
                    continue
                ### adding a marker
                sep_l = [w.strip() if w!=word else '[SEP] {}[SEP]'.format(w.strip()) for w in l]
                ### keeping only reasonable sentences...
                if len(sep_l) > 128:
                    continue
                assert len(sep_l) >= 1
                joint_l = ' '.join(sep_l)
                ### websites are crap...
                if 'http' in joint_l:
                    continue
                ### html too...
                if 'href' in joint_l:
                    continue
                try:
                    all_sentences[word].append(joint_l)
                except KeyError:
                    all_sentences[word] = [joint_l]
    for k, v in all_sentences.items():
        assert len(v) >= 1
    random.seed(11)
    ### sampling 20, so as to avoid bad surprises when extracting...
    all_sentences = {k : random.sample(v, k=min(1000, len(v))) for k, v in all_sentences.items()}
    return all_sentences

def remove_accents(check_str):
    #check_str = check_str.replace('à','').replace('è','').replace('ò','').replace('ú', '').replace('ù', '').replace('é','').replace('á','').replace('ó', '').replace('ö', '').replace('ü', '').replace('ß', '')
    check_str = re.sub('[^a-zA-z]', '', check_str)
    return check_str

def extract_surpr(args, model_card, cases):

    entity_vectors = dict()

    with tqdm() as pbar:
        for dataset, stim_sentences in cases.items():
            entity_vectors[dataset] = dict()
            assert len(stim_sentences) >= 1
            print(dataset)
            print(len(stim_sentences))
            for l_i, l in enumerate(stim_sentences):
                stimulus = l.split()[-1]
                old_l = '{}'.format(l)
                if 'iner' in args.model:
                    l = l.replace(' [SEP] ', '[SEP]')
                print(l)
                inputs = model_card.tokenizer(
                                   l, 
                                   return_tensors="pt",
                                   truncation_strategy='longest_first', 
                                   max_length=int(model_card.max_len*0.75), 
                                   truncation=True,
                                   )
                print(model_card.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
                spans = [i_i for i_i, i in enumerate(inputs['input_ids'].numpy().reshape(-1)) if 
                        i==model_card.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]]
                assert len(spans) == 1
                check_tokens = inputs['input_ids'].numpy().reshape(-1)[spans[0]+1:]
                check_str = ''.join(model_card.tokenizer.convert_ids_to_tokens(check_tokens))
                print(check_str)
                if len(remove_accents(stimulus))!=len(stimulus):
                    if remove_accents(stimulus) not in remove_accents(check_str):
                        print('early skipping: {}'.format(stimulus))
                        continue
                else:
                    if stimulus not in check_str:
                        print('early skipping: {}'.format(stimulus))
                        continue
                del inputs
                l = re.sub(r'\[SEP\]', ' ', l)
                l = re.sub('\s+', r' ', l)
                inputs = model_card.tokenizer(
                                   l, 
                                   return_tensors="pt",
                                   truncation_strategy='longest_first', 
                                   max_length=int(model_card.max_len*0.75), 
                                   truncation=True,
                                   )
                print(model_card.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
                ### Correcting spans
                correction = list(range(1, len(spans)+1))
                if 'll' in args.model or '1.7' in args.model or '2.9' in args.model or 'pt2' in args.model:
                    spans = [max(0, s-c) for s,c in zip(spans, correction)]
                print(model_card.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])[spans[0]:spans[0]+len(check_tokens)])
                ### final checks...
                marker = True
                if len(remove_accents(stimulus))!=len(stimulus):
                    pass
                else:
                    for c_i in range(len(check_tokens)):
                        try:
                            if inputs['input_ids'][0][spans[0]+c_i] != check_tokens[c_i]:
                                marker = False
                        except IndexError:
                            marker = False
                if marker == False:
                    import pdb; pdb.set_trace()
                    print('marker error')
                    continue
                del inputs
                try:
                    inputs = model_card.tokenizer(
                                       l, 
                                       return_tensors="pt",
                                       truncation_strategy='longest_first', 
                                       #max_length=model_card.max_len, 
                                       max_length=int(model_card.max_len*0.75), 
                                       truncation=True,
                                       )
                    if model_card.to_cuda:
                        inputs.to(model_card.cuda_device)
                except RuntimeError:
                    del inputs
                    print('input error')
                    print(l)
                    continue
                print(model_card.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
                try:
                    outputs = model_card.model(
                            **inputs, 
                                    output_attentions=False,
                                    output_hidden_states=True, 
                                    return_dict=True,
                                    )
                except RuntimeError:
                    del inputs
                    print('output error')
                    print(l)
                    continue

                #import pdb; pdb.set_trace()
                #surp = -1 * torch.log2(torch.softmax(outputs.logits.cpu().detach(), -1).squeeze(0)).numpy()
                probs = torch.softmax(outputs.logits.cpu().detach(), dim=-1).squeeze(0).numpy()
                del outputs
                try:
                    assert len(check_tokens) == len(range(spans[0], probs.shape[0]))
                    #assert len(check_tokens) == len(range(spans[0], len(surp)))
                except AssertionError:
                    print('error with {}'.format(old_l))
                    print(check_tokens)
                    print(check_str)
                    print(stimulus)
                    continue
                print(model_card.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])[spans[0]:spans[0]+len(check_tokens)])
                print(model_card.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])[spans[0]])
                #surpr = 0
                tkns = list()
                surpr = list()
                for c_t_i, c_t in enumerate(check_tokens):
                    idx = spans[0]+c_t_i
                    tkns.append(inputs['input_ids'][0][idx])
                    surpr.append(probs[idx][c_t])
                tkns = model_card.tokenizer.convert_ids_to_tokens(tkns)
                print('final tokens: {}'.format(tkns))
                #current_surpr = surpr / len(check_tokens)
                #current_surpr = surpr[-1]
                #current_surpr = surpr[0]
                #current_surpr = numpy.average(surpr)
                current_surpr = -numpy.log2(numpy.prod(surpr))
                #current_surpr = -numpy.log2(numpy.average(surpr))
                #current_surpr = surp[spans[0]][c_t]
                #current_surpr = numpy.average([surp[s_i+1][tok] for s_i, tok in enumerate(inputs['input_ids'][0][1:])])
                ### we take average surprisal
                #current_surpr = -numpy.log2(probs)[spans[0]][check_tokens[0]]
                #current_surpr = -numpy.log2(probs)[spans[0]][check_tokens[0]]
                #current_entropy = -sum([p*numpy.log2(p) for p in probs[spans[0]-1]])
                #current_surpr = surp[spans[0]][check_tokens[0]]
                current_entropy = 'na'
                entity_vectors[dataset][old_l] = (current_surpr, current_entropy)
                pbar.update(1)
                del inputs

    return entity_vectors

def extract_vectors(args, model_card, sentences):

    entity_vectors = dict()
    entity_sentences = dict()

    with tqdm() as pbar:
        for stimulus, stim_sentences in sentences.items():
            print(stimulus)
            #entity_vectors[stimulus] = list()
            #entity_sentences[stimulus] = list()
            assert len(stim_sentences) >= 1
            for l_i, l in enumerate(stim_sentences):
                old_l = '{}'.format(l)
                if 'iner' in args.model:
                    l = l.replace(' [SEP] ', '[SEP]')
                #l = l.replace(' [SEP] ', '[SEP]')
                print(l)

                inputs = model_card.tokenizer(
                                   l, 
                                   return_tensors="pt",
                                   truncation_strategy='longest_first', 
                                   max_length=int(model_card.max_len*0.75), 
                                   truncation=True,
                                   )
                spans = [i_i for i_i, i in enumerate(inputs['input_ids'].numpy().reshape(-1)) if 
                        i==model_card.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]]
                if len(spans) % 2 != 0:
                    print(spans)
                    continue
                try:
                    check_tokens = inputs['input_ids'].numpy().reshape(-1)[spans[0]+1:spans[1]]
                except IndexError:
                    print(check_tokens)
                    continue
                print(check_tokens)
                check_str = ''.join(model_card.tokenizer.convert_ids_to_tokens(check_tokens))
                if stimulus not in check_str:
                    continue
                del inputs
                #print(check_str)
                l = re.sub(r'\[SEP\]', ' ', l)
                l = re.sub('\s+', r' ', l)
                inputs = model_card.tokenizer(
                                   l, 
                                   return_tensors="pt",
                                   truncation_strategy='longest_first', 
                                   #max_length=model_card.max_len, 
                                   max_length=int(model_card.max_len*0.75), 
                                   truncation=True,
                                   )
                if inputs['input_ids'][0].shape[0] > model_card.max_len:
                    print('tokenization error with sentence (length {}):'.format(len(model_card.tokenizer.tokenize(l))))
                    print(l)
                    continue
                ### Correcting spans
                if 'iner' in args.model:
                    start = 0
                else:
                    start = 1
                correction = list(range(start, len(spans)+start))
                spans = [max(0, s-c) for s,c in zip(spans, correction)]
                split_spans = list()
                for i in list(range(len(spans)))[::2]:
                    try:
                        if 'xlm' in model_card.model_name:
                            if 'erta-xl' in model_card.model_name or 'xxl' in model_card.model_name:
                                current_span = (spans[i], spans[i+1])
                            else:
                                current_span = (spans[i]+1, spans[i+1]+1)
                        elif 'llama' in model_card.model_name or 'pt' in args.model:
                            #current_span = (spans[i], spans[i+1]+1)
                            current_span = (spans[i], spans[i+1])
                            #if spans[i] == 0:
                            #    current_span = (spans[i], spans[i+1]+1)
                            #else:
                            #    current_span = (spans[i], spans[i+1])
                        elif 'xglm' in model_card.model_name:
                            if '564m' in args.model:
                                if spans[i] == 0:
                                    #current_span = (spans[i]+2, spans[i+1]+3)
                                    current_span = (spans[i]+2, spans[i+1]+2)
                                else:
                                    #current_span = (spans[i]+1, spans[i+1]+2)
                                    current_span = (spans[i]+1, spans[i+1]+1)
                            else:
                                if spans[i] == 0:
                                    #current_span = (spans[i]+1, spans[i+1]+2)
                                    current_span = (spans[i]+1, spans[i+1]+1)
                                else:
                                    #current_span = (spans[i], spans[i+1]+1)
                                    current_span = (spans[i], spans[i+1])
                        #error = inputs['input_ids'][0][current_span[0]:current_span[1]]
                        #error_str = ''.join(model_card.tokenizer.convert_ids_to_tokens(error))
                        #print(error_str)
                    except IndexError:
                        print('there was a mistake with: {}'.format(l))
                        continue
                    ### final checks...
                    marker = True
                    for c_i in range(len(check_tokens)):
                        try:
                            if inputs['input_ids'][0][current_span[0]+c_i] != check_tokens[c_i]:
                                marker = False
                        except IndexError:
                            marker = False
                    if marker == False:
                        continue
                    if len(range(current_span[0], current_span[1])) > len(check_tokens):
                        current_span[1] = current_span[0]+len(check_tokens)+1
                    split_spans.append(current_span)
                del inputs
                ### taking only right-most mention
                if len(split_spans) > 1:
                    split_spans = [split_spans[-1]]
                try:
                    inputs = model_card.tokenizer(
                                       l, 
                                       return_tensors="pt",
                                       truncation_strategy='longest_first', 
                                       #max_length=model_card.max_len, 
                                       max_length=int(model_card.max_len*0.75), 
                                       truncation=True,
                                       )
                    if model_card.to_cuda:
                        inputs.to(model_card.cuda_device)
                except RuntimeError:
                    del inputs
                    print('input error')
                    print(l)
                    continue
                try:
                    outputs = model_card.model(
                            **inputs, 
                                    output_attentions=False,
                                    output_hidden_states=True, 
                                    return_dict=True,
                                    )
                except RuntimeError:
                    del inputs
                    print('output error')
                    print(l)
                    continue

                hidden_states = numpy.array([s[0].cpu().detach().numpy() for s in outputs['hidden_states']])
                del outputs
                #last_hidden_states = numpy.array([k.detach().numpy() for k in outputs['hidden_states']])[2:6, 0, :]
                for beg, end in split_spans:
                    ### If there are less than two tokens that must be a mistake
                    to_remove = model_card.tokenizer.convert_tokens_to_ids(['_'])+model_card.tokenizer.convert_tokens_to_ids([' '])
                    print(to_remove)
                    if inputs['input_ids'][0][beg] in to_remove:
                        beg += 1
                    if len(inputs['input_ids'][0][beg:end]) < 1:
                        continue
                    tkns = model_card.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][beg:end])
                    #if stimulus not in ''.join(tkns):
                    #    continue
                    print(tkns)
                    mention = hidden_states[:-1, beg:end, :]
                    mention = numpy.average(mention, axis=1)
                    #mention = mention[layer_start:layer_end, :]

                    #mention = numpy.average(mention, axis=0)
                    assert mention.shape == (model_card.n_layers, model_card.required_shape, )
                    try:
                        entity_vectors[stimulus].append(mention)
                        entity_sentences[stimulus].append(old_l)
                    except KeyError:
                        entity_vectors[stimulus] = [mention]
                        entity_sentences[stimulus] = [old_l]
                    pbar.update(1)
                del inputs
    for k, v in entity_vectors.items():
        for mention in v:
            assert mention.shape == (model_card.n_layers, model_card.required_shape, )

    return entity_vectors, entity_sentences

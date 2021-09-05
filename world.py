import paddle
import paddle.nn as nn
import numpy as np
import paddlehub as hub
from paddlenlp.transformers import GPTForPretraining, GPTChineseTokenizer
import queue

class Model():
    def __init__(self):
        self.model = GPTForPretraining.from_pretrained('gpt-cpm-small-cn-distill')
        # self.model = GPT2ForPretraining.from_pretrained('gpt2-base-cn')
        self.model.eval()
        self.tokenizer = GPTChineseTokenizer.from_pretrained('gpt-cpm-small-cn-distill')
        # self.tokenizer = GPT2ChineseTokenizer.from_pretrained('gpt2-base-cn')
        self.tokenizer.encode('.')


    def top_k_top_p_filtering(self, logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
        top_k = min(top_k, logits.shape[-1])  # Safety check
        logits_np = logits.numpy()
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits_np < np.sort(logits_np)[-top_k]
            logits_np[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits = paddle.sort(logits, descending=True)
            sorted_indices = paddle.argsort(logits, descending=True).numpy()
            cumulative_probs = paddle.cumsum(paddle.nn.functional.softmax(
                sorted_logits, axis=-1), axis=-1).numpy()

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[...,
                                        1:] = sorted_indices_to_remove[..., :-1]
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits_np[indices_to_remove] = filter_value

        return paddle.to_tensor(logits_np)

    def sample(self, text, max_len=32, end_word='。', repitition_penalty=1.0, temperature=1.0, top_p=0.9):
        with paddle.no_grad():
            # 终止标志
            if end_word is not None:
                stop_id = self.tokenizer.encode(end_word)
                if 'input_ids' in stop_id:
                    stop_id = self.tokenizer._convert_token_to_id(end_word)
                    
            ids = self.tokenizer.encode(text)
            if 'input_ids' in ids:
                ids = ids['input_ids']
                input_ids = paddle.to_tensor(ids).unsqueeze(0)
            else:
                input_ids = paddle.to_tensor(np.array(ids).reshape(1, -1).astype('int64'))
            output, cached_kvs = self.model(input_ids, use_cache=True)
            next_token_logits = output[0, -1, :]
            for id in set(ids):
                next_token_logits[id] /= repitition_penalty
            next_token_logits = next_token_logits / temperature
            filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=0, top_p=1.0)
            next_token = paddle.multinomial(paddle.nn.functional.softmax(filtered_logits, axis=-1), num_samples=1).numpy()
            ids += [int(next_token)]

            for i in range(max_len):
                input_id = paddle.to_tensor(np.array([next_token]).reshape(1, -1).astype('int64'))
                output, cached_kvs = self.model(input_id, use_cache=True, cache=cached_kvs)
                next_token_logits = output[0, -1, :]
                for id in set(ids):
                    next_token_logits[id] /= repitition_penalty
                next_token_logits = next_token_logits / temperature
                filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=0, top_p=1.0)
                next_token = paddle.multinomial(paddle.nn.functional.softmax(filtered_logits, axis=-1), num_samples=1).numpy()
                ids += [int(next_token)]

                # 根据终止标志停止预测
                if (end_word is not None) and (int(next_token) == stop_id):
                    break

            return self.tokenizer.convert_ids_to_string(ids)


temp_event = [
    '现在你来到一个城市之中，丧尸在街道上嘶吼，你好像是来到了是浣熊市，而核弹将在一小时后降临。',
    '历经千辛万苦，你来到一个地下实验室，旁边冰冷的警报声在提示着你T病毒已经泄露，一管病毒试剂摆在你面前。',
    '你来到了三体世界，你身上沾染的病毒不小心泄露到了这个世界，现在整个世界都在面临生化危机。',
    '无处可逃的你跑到了“自然选择”号飞船的队伍之中，你意识到三体人的舰队即将抵达，章北海会劫持飞船逃离太阳系。',
]

class World():
    def __init__(self):
        self.init = False
        self.model = Model()

    def sample(self, sentence):
        outputs = self.model.sample(
            sentence, # 输入文本
            max_len=64, # 最大生成文本的长度
            #end_word=None, # 终止符号
            end_word='。',
        )
        outputs = outputs.replace(',', '，').replace(sentence,'')  
        return outputs

    def start(self):
        self.context_list = [
            '冰冷，抖动……醒来的瞬间，你猛的从地面跳了起来，惊慌的看向四周，脑海里的办公室环境和眼前的环境瞬间出现了混淆，几秒之后你从混淆里清醒过来。',
        ]
        self.lastPoint = 1
        self.init = True
        self.worlds_event = queue.Queue(maxsize=0)
        for i in temp_event:
            i = i.replace('\n','')
            self.worlds_event.put(i)
        event = self.worlds_event.get()
        self.context_list.append(event)
        return ''.join(self.context_list)

    def receive(self, user_input):
        if self.worlds_event.empty():
            print('你的冒险已经结束')
            return '你的冒险已经结束'
        user_output = ''
        inputs = user_input + '，'
        self.context_list.append(inputs)
        context = ''.join(self.context_list[-2:])
        user_output = self.sample(context)
        self.context_list.append(user_output)
        self.lastPoint += 1

        if self.lastPoint % 4 == 0:
            if self.worlds_event.empty():
                print(user_output, '结束啦')
                return '你的冒险已经结束'
            event = self.worlds_event.get()
            self.context_list.append(event)
            user_output = user_output + event
        
        print(user_output)
        return user_output

if __name__ == '__main__':
    world = World()
    print(world.start())
    while True:
        ans = input()
        if world.receive(ans) == '你的冒险已经结束':
            break
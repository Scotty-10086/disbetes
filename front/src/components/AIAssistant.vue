<script setup>
import { ref, watch } from 'vue';
import { marked } from 'marked';
import DOMPurify from 'dompurify';

const props = defineProps({
  predictionResult: {
    type: Object,
    default: () => ({})
  }
});

const chatMessages = ref([]);
const userInput = ref('');
const isLoading = ref(false);

// 监听预测结果变化
watch(() => props.predictionResult, async (newVal) => {
  if (Object.keys(newVal).length > 0) {
    await analyzeResults(newVal);
  }
});

// 分析预测结果并给出建议
async function analyzeResults(results) {
  isLoading.value = true;
  try {
    const response = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'deepseek-r1:8b',
        stream: true,
        prompt: `作为一个医疗助手，请分析以下糖尿病预测结果并给出专业建议：
预测结果：${results.outcome ? '存在糖尿病风险' : '糖尿病风险较低'}
准确率：${(results.accuracy * 100).toFixed(2)}%
精确率：${(results.precision * 100).toFixed(2)}%
召回率：${(results.recall * 100).toFixed(2)}%`
      })
    });

    if (!response.ok) throw new Error('请求失败');
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let currentMessage = {
      role: 'assistant',
      content: '',
      htmlContent: ''
    };
    chatMessages.value.push(currentMessage);

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n').filter(line => line.trim());
      
      for (const line of lines) {
        try {
          const data = JSON.parse(line);
          if (data.response) {
            currentMessage.content += data.response;
            // 使用Vue的响应式更新
            chatMessages.value[chatMessages.value.length - 1] = {
              ...currentMessage,
              htmlContent: DOMPurify.sanitize(marked(currentMessage.content))
            };
          }
        } catch (e) {
          console.error('解析响应数据出错:', e);
        }
      }
    }
  } catch (error) {
    console.error('分析结果出错:', error);
    chatMessages.value.push({
      role: 'assistant',
      content: '抱歉，分析结果时出现错误，请稍后重试。'
    });
  } finally {
    isLoading.value = false;
  }
}

// 发送用户消息
async function sendMessage() {
  if (!userInput.value.trim()) return;

  const message = userInput.value;
  chatMessages.value.push({
    role: 'user',
    content: message
  });
  userInput.value = '';
  
  isLoading.value = true;
  try {
    const response = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'deepseek-r1:8b',
        stream: true,
        prompt: message
      })
    });

    if (!response.ok) throw new Error('请求失败');
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let currentMessage = {
      role: 'assistant',
      content: '',
      htmlContent: ''
    };
    chatMessages.value.push(currentMessage);

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n').filter(line => line.trim());
      
      for (const line of lines) {
        try {
          const data = JSON.parse(line);
          if (data.response) {
            currentMessage.content += data.response;
            // 使用Vue的响应式更新
            chatMessages.value[chatMessages.value.length - 1] = {
              ...currentMessage,
              htmlContent: DOMPurify.sanitize(marked(currentMessage.content))
            };
          }
        } catch (e) {
          console.error('解析响应数据出错:', e);
        }
      }
    }
  } catch (error) {
    console.error('发送消息出错:', error);
    chatMessages.value.push({
      role: 'assistant',
      content: '抱歉，处理消息时出现错误，请稍后重试。'
    });
  } finally {
    isLoading.value = false;
  }
}
</script>

<template>
  <div class="ai-assistant">
    <h3>AI 健康助手</h3>
    
    <div class="chat-container">
      <div class="messages" ref="messagesContainer">
        <div
          v-for="(message, index) in chatMessages"
          :key="index"
          :class="['message', message.role]"
        >
          <div v-if="message.htmlContent" v-html="message.htmlContent" class="markdown-content"></div>
          <div v-else>{{ message.content }}</div>
        </div>
      </div>
      
      <div class="input-container">
        <input
          v-model="userInput"
          type="text"
          placeholder="输入您的问题..."
          @keyup.enter="sendMessage"
          :disabled="isLoading"
        >
        <button
          @click="sendMessage"
          :disabled="isLoading || !userInput.trim()"
        >
          发送
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.markdown-content :deep(p) {
  margin: 8px 0;
}

.markdown-content :deep(ul), .markdown-content :deep(ol) {
  padding-left: 20px;
  margin: 8px 0;
}

.markdown-content :deep(h1), .markdown-content :deep(h2), .markdown-content :deep(h3),
.markdown-content :deep(h4), .markdown-content :deep(h5), .markdown-content :deep(h6) {
  margin: 16px 0 8px;
  font-weight: 600;
}

.markdown-content :deep(code) {
  background-color: #f6f8fa;
  padding: 2px 4px;
  border-radius: 4px;
  font-family: monospace;
}

.markdown-content :deep(pre) {
  background-color: #f6f8fa;
  padding: 12px;
  border-radius: 4px;
  overflow-x: auto;
}

.markdown-content :deep(blockquote) {
  border-left: 4px solid #dfe2e5;
  padding-left: 16px;
  margin: 8px 0;
  color: #6a737d;
}
.ai-assistant {
  width: 100%;
  max-width: 600px;
  margin: 20px auto;
  padding: 20px;
  background-color: #ffffff;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.chat-container {
  margin-top: 20px;
}

.messages {
  height: 300px;
  overflow-y: auto;
  padding: 10px;
  border: 1px solid #e8e8e8;
  border-radius: 4px;
  margin-bottom: 10px;
}

.message {
  margin: 8px 0;
  padding: 8px 12px;
  border-radius: 4px;
  max-width: 80%;
  word-wrap: break-word;
}

.message.user {
  background-color: #e3f2fd;
  margin-left: auto;
}

.message.assistant {
  background-color: #f5f5f5;
  margin-right: auto;
}

.input-container {
  display: flex;
  gap: 10px;
}

input {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  outline: none;
}

input:focus {
  border-color: #409eff;
}

button {
  padding: 8px 16px;
  background-color: #409eff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.3s;
}

button:hover {
  background-color: #66b1ff;
}

button:disabled {
  background-color: #a0cfff;
  cursor: not-allowed;
}
</style>
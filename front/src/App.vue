<script setup>
import { ref } from 'vue';
import DataForm from './components/DataForm.vue';
import ResultDisplay from './components/ResultDisplay.vue';
import AIAssistant from './components/AIAssistant.vue';

const predictionResult = ref({});

const handleSubmit = async (formData) => {
  try {
    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(formData),
    });

    if (!response.ok) {
      throw new Error('预测请求失败');
    }

    const result = await response.json();
    predictionResult.value = result;
  } catch (error) {
    console.error('预测出错:', error);
    alert('预测失败，请稍后重试');
  }
};
</script>

<template>
  <div class="app">
    <header>
      <h1>糖尿病风险预测系统</h1>
    </header>
    <main>
      <DataForm @submit-data="handleSubmit" />
      <ResultDisplay :prediction-result="predictionResult" />
      <AIAssistant :prediction-result="predictionResult" />
    </main>
  </div>
</template>

<style scoped>
.app {
  min-height: 100vh;
  background-color: #f0f2f5;
  padding: 20px;
}

header {
  text-align: center;
  margin-bottom: 40px;
}

h1 {
  color: #2c3e50;
  font-size: 2.5em;
  margin: 0;
  padding: 20px 0;
}

main {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
  justify-content: center;
}
</style>

<script setup>
import { ref, watch } from 'vue';

const props = defineProps({
  predictionResult: {
    type: Object,
    default: () => ({})
  }
});

const showResult = ref(false);
const validResult = ref(false);

// 验证数据的有效性
const validateResult = (result) => {
  if (!result || typeof result !== 'object') return false;
  const requiredFields = ['outcome', 'accuracy', 'precision', 'recall', 'f1_score'];
  return requiredFields.every(field => field in result && result[field] !== null);
};

// 监听预测结果变化
watch(() => props.predictionResult, (newVal) => {
  console.log('预测结果变化：', newVal);
  validResult.value = validateResult(newVal);
  showResult.value = validResult.value && Object.keys(newVal).length > 0;
});
</script>

<template>
  <div class="result-display" v-if="showResult && validResult">
    <h3>预测结果</h3>
    <div class="result-content">
      <div class="result-item">
        <span class="label">预测结果：</span>
        <span class="value" :class="predictionResult.outcome ? 'positive' : 'negative'">
          {{ predictionResult.outcome ? '存在糖尿病风险' : '糖尿病风险较低' }}
        </span>
      </div>

      <div class="result-item">
        <span class="label">准确率：</span>
        <span class="value">{{ (predictionResult.accuracy * 100).toFixed(2) }}%</span>
      </div>

      <div class="model-details">
        <h4>模型评估详情</h4>
        <div class="metrics">
          <div class="metric-item">
            <span class="metric-label">精确率：</span>
            <span class="metric-value">{{ (predictionResult.precision * 100).toFixed(2) }}%</span>
          </div>
          <div class="metric-item">
            <span class="metric-label">召回率：</span>
            <span class="metric-value">{{ (predictionResult.recall * 100).toFixed(2) }}%</span>
          </div>
          <div class="metric-item">
            <span class="metric-label">F1分数：</span>
            <span class="metric-value">{{ (predictionResult.f1_score * 100).toFixed(2) }}%</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.result-display {
  max-width: 600px;
  margin: 20px auto;
  padding: 20px;
  background-color: #f8f9fa;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

h3 {
  color: #2c3e50;
  margin-bottom: 20px;
  text-align: center;
}

.result-content {
  padding: 15px;
}

.result-item {
  margin-bottom: 15px;
  display: flex;
  align-items: center;
}

.label {
  font-weight: bold;
  color: #34495e;
  min-width: 100px;
}

.value {
  font-size: 1.1em;
}

.positive {
  color: #e74c3c;
  font-weight: bold;
}

.negative {
  color: #2ecc71;
  font-weight: bold;
}

.model-details {
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px solid #ddd;
}

h4 {
  color: #2c3e50;
  margin-bottom: 15px;
}

.metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
}

.metric-item {
  background-color: white;
  padding: 10px;
  border-radius: 4px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.metric-label {
  color: #7f8c8d;
  font-size: 0.9em;
}

.metric-value {
  display: block;
  color: #2c3e50;
  font-size: 1.2em;
  font-weight: bold;
  margin-top: 5px;
}
</style>
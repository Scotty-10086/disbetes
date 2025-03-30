<script setup>
import { ref } from 'vue';

const formData = ref({
  Pregnancies: '',
  Glucose: '',
  BloodPressure: '',
  SkinThickness: '',
  Insulin: '',
  BMI: '',
  DiabetesPedigree: '',
  Age: ''
});

const emit = defineEmits(['submit-data']);

const submitForm = () => {
  emit('submit-data', formData.value);
};

const resetForm = () => {
  Object.keys(formData.value).forEach(key => {
    formData.value[key] = '';
  });
};

const loadHealthyExample = () => {
  formData.value = {
    Pregnancies: '2',
    Glucose: '85',
    BloodPressure: '70',
    SkinThickness: '18',
    Insulin: '80',
    BMI: '22.5',
    DiabetesPedigree: '0.2',
    Age: '28'
  };
};
</script>

<template>
  <div class="data-form">
    <h2>糖尿病风险预测 - 数据输入</h2>
    <form @submit.prevent="submitForm">
      <div class="form-group">
        <label for="pregnancies">怀孕次数：</label>
        <input
          id="pregnancies"
          v-model="formData.Pregnancies"
          type="number"
          min="0"
          required
        >
      </div>

      <div class="form-group">
        <label for="glucose">血糖浓度：</label>
        <input
          id="glucose"
          v-model="formData.Glucose"
          type="number"
          min="0"
          required
        >
      </div>

      <div class="form-group">
        <label for="bloodPressure">血压(mm Hg)：</label>
        <input
          id="bloodPressure"
          v-model="formData.BloodPressure"
          type="number"
          min="0"
          required
        >
      </div>

      <div class="form-group">
        <label for="skinThickness">皮褶厚度(mm)：</label>
        <input
          id="skinThickness"
          v-model="formData.SkinThickness"
          type="number"
          min="0"
          required
        >
      </div>

      <div class="form-group">
        <label for="insulin">胰岛素(mu U/ml)：</label>
        <input
          id="insulin"
          v-model="formData.Insulin"
          type="number"
          min="0"
          required
        >
      </div>

      <div class="form-group">
        <label for="bmi">体重指数(BMI)：</label>
        <input
          id="bmi"
          v-model="formData.BMI"
          type="number"
          min="0"
          step="0.1"
          required
        >
      </div>

      <div class="form-group">
        <label for="diabetesPedigree">糖尿病家族史：</label>
        <input
          id="diabetesPedigree"
          v-model="formData.DiabetesPedigree"
          type="number"
          min="0"
          step="0.01"
          required
        >
      </div>

      <div class="form-group">
        <label for="age">年龄：</label>
        <input
          id="age"
          v-model="formData.Age"
          type="number"
          min="0"
          required
        >
      </div>

      <div class="button-group">
        <button type="submit" class="submit-btn">提交预测</button>
        <button type="button" @click="resetForm" class="reset-btn">重置表单</button>
        <button type="button" @click="loadHealthyExample" class="example-btn">加载示例数据</button>
      </div>
    </form>
  </div>
</template>

<style scoped>
.data-form {
  max-width: 500px;
  margin: 0 auto;
  padding: 20px;
}

h2 {
  color: #2c3e50;
  text-align: center;
  margin-bottom: 30px;
}

.form-group {
  margin-bottom: 20px;
}

label {
  display: block;
  margin-bottom: 5px;
  color: #34495e;
  font-weight: bold;
}

input {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  font-size: 14px;
  transition: border-color 0.2s;
}

input:focus {
  border-color: #409eff;
  outline: none;
}

.button-group {
  display: flex;
  gap: 10px;
  margin-top: 30px;
}

button {
  flex: 1;
  padding: 10px 20px;
  border: none;
  border-radius: 4px;
  font-size: 14px;
  cursor: pointer;
  transition: background-color 0.3s;
}

.submit-btn {
  background-color: #409eff;
  color: white;
}

.submit-btn:hover {
  background-color: #66b1ff;
}

.reset-btn {
  background-color: #909399;
  color: white;
}

.reset-btn:hover {
  background-color: #a6a9ad;
}

.example-btn {
  background-color: #67c23a;
  color: white;
}

.example-btn:hover {
  background-color: #85ce61;
}
</style>
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "py_code"))
sys.path.insert(0, str(Path(__file__).parent))

from mag_nav import Solenoid, GaussianNoise, Scenario, SensorConfig, line
from make_dataset import features_from_ab
from catboost import CatBoostRegressor

st.set_page_config(page_title="ML Навигация (Multi-Beacon)", layout="wide")
st.title("🧭 ML-навигация: универсальная модель для разных маяков")
st.markdown("Модель обучена на 4 типах соленоидов и работает с любым из них")

# --- Боковая панель ---
st.sidebar.header("⚙️ Параметры маяка")
beacon_type = st.sidebar.selectbox(
    "Тип маяка",
    ["small", "medium", "large", "tiny"],
    help="Выберите конфигурацию соленоида"
)

beacon_configs = {
    "small": {"n_turns": 200, "current": 0.5, "radius": 0.04, "length": 0.08},
    "medium": {"n_turns": 500, "current": 1.0, "radius": 0.06, "length": 0.12},
    "large": {"n_turns": 1000, "current": 2.0, "radius": 0.08, "length": 0.16},
    "tiny": {"n_turns": 100, "current": 0.3, "radius": 0.03, "length": 0.05},
}

st.sidebar.header("⚙️ Параметры симуляции")
sigma = st.sidebar.slider("Уровень шума (Тл)", 0.0, 5e-9, 1e-9, 1e-10, format="%.1e")
n_points = st.sidebar.slider("Точек траектории", 50, 500, 150)

# --- Загрузка модели ---
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "results" / "multi_beacon" / "model_multi.cbm"
    if model_path.exists():
        model = CatBoostRegressor()
        model.load_model(str(model_path))
        return model
    else:
        st.error("❌ Модель не найдена!")
        return None

model = load_model()

# --- Кнопка генерации ---
if st.button("🔄 Сгенерировать траекторию", type="primary"):
    if model is None:
        st.stop()

    with st.spinner("Генерация и предсказание..."):
        config = beacon_configs[beacon_type]
        
        # 1. Источник
        source = Solenoid(
            origin=np.array([0.0, 0.0, 0.0]),
            **config,
            n_quad=96
        )
        b0 = source.dipole_constant

        # 2. Траектория
        start = np.array([1.0, 0.8, 0.6])
        end = np.array([0.15, 0.12, 0.10])
        trajectory = line(start, end, n_points)

        # 3. Шум
        noise = GaussianNoise(sigma=sigma)
        sensor = SensorConfig(
            samples_per_period=64, n_periods=4, noise=noise,
            demod_method="lstsq", with_dc=True
        )

        # 4. Прогон
        scenario = Scenario(
            source=source, trajectory=trajectory, sensor=sensor,
            inject_at="samples", seed=42
        )
        result = scenario.run()

        # 5. Извлекаем признаки
        p_true = result.true_positions
        p_dip = result.estimated_positions
        r_dip = np.linalg.norm(p_dip, axis=1)
        
        X_base, _, valid = features_from_ab(result.a_meas, result.b_meas, b0)
        
        # Добавляем параметры маяка
        beacon_params = np.array([
            np.log10(config["n_turns"]),
            np.log10(config["current"]),
            np.log10(config["radius"]),
            np.log10(config["length"]),
            np.log10(b0),
        ])
        beacon_params_expanded = np.tile(beacon_params, (X_base.shape[0], 1))
        X_extended = np.hstack([X_base, beacon_params_expanded])
        
        # Предсказываем
        delta = np.zeros_like(p_dip)
        if valid.any():
            delta[valid] = model.predict(X_extended[valid])
        
        p_ml = p_dip + r_dip[:, None] * delta

        # --- Визуализация ---
        st.subheader(f"📊 3D-траектория (маяк: {beacon_type})")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot(p_true[:, 0], p_true[:, 1], p_true[:, 2], 'k-', linewidth=3, label="Истинная")
        ax.plot(p_dip[:, 0], p_dip[:, 1], p_dip[:, 2], 'r--', alpha=0.7, linewidth=2, label="Дипольная")
        ax.plot(p_ml[:, 0], p_ml[:, 1], p_ml[:, 2], 'g-', linewidth=2.5, label="CatBoost (ML)")
        
        ax.scatter([0], [0], [0], color='black', s=150, marker='*', label="Маяк", zorder=10)
        
        ax.set_xlabel("X, м")
        ax.set_ylabel("Y, м")
        ax.set_zlabel("Z, м")
        ax.set_title(f"Траектория (σ = {sigma:.1e} Тл, маяк: {beacon_type})")
        ax.legend()
        
        st.pyplot(fig)

        # --- Метрики ---
        st.subheader("📈 Метрики ошибки")
        err_dip = np.linalg.norm(p_dip - p_true, axis=1)
        err_ml = np.linalg.norm(p_ml - p_true, axis=1)
        
        rms_dip = np.sqrt(np.mean(err_dip**2))
        rms_ml = np.sqrt(np.mean(err_ml**2))
        improvement = rms_dip / rms_ml if rms_ml > 0 else float('inf')

        col1, col2, col3 = st.columns(3)
        col1.metric("RMS (Диполь)", f"{rms_dip:.4f} м")
        col2.metric("RMS (CatBoost)", f"{rms_ml:.4f} м")
        col3.metric("Улучшение", f"{improvement:.2f}x")
        
        # Информация о маяке
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📡 Параметры выбранного маяка")
        st.sidebar.markdown(f"- **Витков:** {config['n_turns']}")
        st.sidebar.markdown(f"- **Ток:** {config['current']} А")
        st.sidebar.markdown(f"- **Радиус:** {config['radius']} м")
        st.sidebar.markdown(f"- **Длина:** {config['length']} м")
        st.sidebar.markdown(f"- **B0:** {b0:.2e} Тл·м³")
"""
Script para reentrenar los modelos con la versión actual de scikit-learn
Soluciona problemas de compatibilidad entre versiones
"""

from ml_models import train_and_save_model

if __name__ == "__main__":
    print("=" * 60)
    print("Reentrenando modelos con la versión actual de scikit-learn")
    print("=" * 60)
    
    model, results = train_and_save_model()
    
    print("\n✅ Modelos reentrenados y guardados exitosamente!")
    print(f"   Train accuracy: {results['train_accuracy']:.2%}")
    print(f"   Test accuracy: {results['test_accuracy']:.2%}")

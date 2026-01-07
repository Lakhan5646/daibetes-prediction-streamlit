plt.savefig("../reports/confusion_matrix.png")
plt.savefig("../reports/heatmap.png")
importance_df.to_csv("../reports/feature_importance.csv", index=False)

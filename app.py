if st.button("Classify Traffic"):
    with st.spinner("Processing and classifying..."):
        if 'Label' in df.columns:
            y = df['Label']
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # Train model
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Attach prediction results
            df['Predicted_Class'] = model.predict(X_scaled)
            st.success("Classification complete using training data!")

            # Classification Results
            st.subheader("Classification Results")
            st.write(df[['Length', 'Protocol', 'Source', 'Destination', 'Predicted_Class']].head(10))

            # Confusion Matrix
            cm = confusion_matrix(y_test, predictions)
            labels = ['Low', 'Medium', 'High']  # Adjust if label encoding is different
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.subheader("Confusion Matrix")
            st.pyplot(fig)

            # RFECV Feature Selection
            st.subheader("Feature Selection via RFECV")
            rfecv_model = DecisionTreeClassifier(random_state=42)
            rfecv = RFECV(estimator=rfecv_model, scoring='accuracy', cv=5, n_jobs=-1)
            pipeline = Pipeline(steps=[('feature_selection', rfecv), ('classification', rfecv_model)])
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
            scores = cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
            st.write(f"**RFECV Accuracy:** {mean(scores):.3f} (±{std(scores):.3f})")

            # RFE for Feature Selection
            st.subheader("Feature Importance via RFE")
            original_features = ['Length', 'Protocol_encoded', 'Source_encoded', 'Destination_encoded']
            rfe_selector = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=3)
            rfe_selector.fit(X_train, y_train)
            for i, feature in enumerate(original_features):
                st.write(f"Feature: {feature:>20} | Selected: {rfe_selector.support_[i]} | Rank: {rfe_selector.ranking_[i]}")
        
        else:
            st.warning("No 'Label' column found. Model will predict without training. Results may be invalid.")
            try:
                predictions = model.predict(X_scaled)
                df['Predicted_Class'] = predictions
            except Exception as e:
                st.error(f"Model couldn't classify without training: {e}")
                st.stop()

        # Prediction distribution
        st.subheader("Distribution of Predictions")
        st.bar_chart(df['Predicted_Class'].value_counts())

        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name='classified_traffic.csv',
            mime='text/csv',
        )

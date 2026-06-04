pipeline {
    agent any

    options {
        timestamps()
        // For avoiding double checkout
        skipDefaultCheckout()
    }
parameters {
    booleanParam(name: 'RUN_TRAINING', defaultValue: false, description: 'Run training stage?')
    booleanParam(name: 'RUN_EVALUATION',  defaultValue: true,  description: 'Run evaluation?')
    booleanParam(name: 'REGISTER_MODEL',  defaultValue: false, description: 'Push to model registry?')
    choice(name: 'DATASET', choices: ['small', 'medium', 'full'], description: 'Dataset size?')
    }



    stages {
        stage('Clean Workspace') {
            steps {
                cleanWs()
                checkout scm

            }
        }
        stage('Checkout') {
            steps {
                echo "Using branch: ${env.BRANCH_NAME ?: env.GIT_BRANCH ?: 'unknown'}"
                checkout scm
                sh 'git rev-parse --short HEAD'
            }
        }


        // opbygning af docker image
        stage('Build Docker Image') {
            steps {

                sh '''
                    docker --version
                    GIT_COMMIT=$(git rev-parse HEAD)
                    echo "Building docker image from SHA: $GIT_COMMIT"
                    echo "Cutting SHA to short..."
                    SHORT_SHA=$(echo "$GIT_COMMIT" | cut -c1-7)
                    echo "Short SHA: $SHORT_SHA"

                    docker build \
                    -t mlops_project_tests:$BUILD_NUMBER \
                    -t mlops_project_tests:$SHORT_SHA \
                    .
                '''
            }
        }

        stage('Run Unit Tests (pytest)') {
            steps {
                sh 'docker run --rm mlops_project_tests:$BUILD_NUMBER python -m pytest -q'
            }
        }

        stage('Pull Data with DVC'){
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'minio_ass',
                    usernameVariable: 'AWS_ACCESS_KEY_ID',
                    passwordVariable: 'AWS_SECRET_ACCESS_KEY' // pragma: allowlist secret
                    )]) {
                sh '''
                    docker run --rm \
                    -e AWS_ACCESS_KEY_ID \
                    -e AWS_SECRET_ACCESS_KEY \
                    -v "$PWD:/app" \
                    -w /app \
                    mlops_project_tests:$BUILD_NUMBER \
                    dvc pull -v
                '''
            }
            }
        }
        stage('Training Model') {
            when { expression { return params.RUN_TRAINING } }
            steps {
                sh '''
                    mkdir -p outputs
                    GIT_COMMIT=$(git rev-parse HEAD)
                    GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

                    docker run --rm \
                        --gpus all \
                        -v "$PWD:/app" \
                        -w /app \
                        -e CUDA_HOME=/usr/local/cuda \
                        -e MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
                        -e MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME}" \
                        -e GIT_COMMIT="$GIT_COMMIT" \
                        -e GIT_BRANCH="$GIT_BRANCH" \
                        -e JOB_NAME="${JOB_NAME}" \
                        -e BUILD_NUMBER="${BUILD_NUMBER}" \
                        -e BUILD_URL="${BUILD_URL}" \
                        mlops_project_tests:$BUILD_NUMBER \
                        deepspeed --num_gpus=1 train.py
                '''
            }
            post {
                always {
                    archiveArtifacts artifacts: 'outputs/**', fingerprint: true, allowEmptyArchive: true
                }
            }
        }
        
        //stage('Detect Drift') {
        //    when { expression { return params.RUN_EVALUATION } }
        //    steps {
        //        sh '''
        //            docker run --rm \
        //                -v "$PWD:/app" \
        //                -w /app \
        //                -e MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
        //                -e MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME}" \
        //                mlops_project_tests:$BUILD_NUMBER \
        //                python detect_drift.py
        //        '''
        //    }
        //}



        stage('Cleanup'){
            steps {
            // cleanup
                    sh '''
                        GIT_COMMIT=$(git rev-parse HEAD)
                        SHORT_SHA=$(echo "$GIT_COMMIT" | cut -c1-7)
                        docker image rm -f mlops_project_tests:$BUILD_NUMBER mlops_project_tests:$SHORT_SHA || true

                    '''
            }
                    }
    }
}

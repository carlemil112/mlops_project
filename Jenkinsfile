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


        // Docker image Registry push
        stage('Push Docker Image') {
            steps {
                sh '''
                    GIT_COMMIT=$(git rev-parse HEAD)
                    SHORT_SHA=$(echo "$GIT_COMMIT" | cut -c1-7)

                    docker tag mlops_project_tests:$BUILD_NUMBER $REGISTRY_URL/rasmil112:$SHORT_SHA
                    docker tag mlops_project_tests:$BUILD_NUMBER $REGISTRY_URL/rasmil112:latest

                    docker push $REGISTRY_URL/rasmil112:$SHORT_SHA
                    docker push $REGISTRY_URL/rasmil112:latest
                '''
            }
        }




        stage('Run Unit Tests (pytest)') {
            steps {
                sh 'docker run --rm mlops_project_tests:$BUILD_NUMBER python -m pytest -q --cov=. --cov-report=term-missing'
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

                    curl -X POST http://172.24.198.42:8000/metrics/stage \
                        -H "Content-Type: application/json" \
                        -d '{"stage":"dvc_pull","success":1}' || true
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
                        deepspeed --num_gpus=2 train.py



                        curl -X POST http://172.24.198.42:8000/metrics/stage \
                            -H "Content-Type: application/json" \
                            -d '{"stage":"training","success":1}' || true
                '''
            }
            post {
                always {
                    archiveArtifacts artifacts: 'outputs/**', fingerprint: true, allowEmptyArchive: true
                }
            }
        }

        stage('Testing gpu workers') {
            steps{
                sh'''
                python -c "import torch; print('cuda devices:', torch.cuda.device_count())"
                nvidia-smi
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

        stage('Post quantization of model') {
            when { expression { return params.RUN_EVALUATION } }
            steps {
                sh '''
                    docker run --rm \
                        -v "$PWD:/app" \
                        -w /app \
                        -e MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
                        mlops_project_tests:$BUILD_NUMBER \
                        python convert_model.py
                '''
            }
        }


        stage('Evaluate Model') {
            when { expression { return params.RUN_EVALUATION } }
            steps {
                sh '''
                    docker run --rm \
                        -v "$PWD:/app" \
                        -w /app \
                        -e MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
                        mlops_project_tests:$BUILD_NUMBER \
                        python inference.py
                '''
            }
        }

        stage('Merge to Main') {
            when {
                expression {
                    def branch = sh(script: 'git name-rev --name-only HEAD', returnStdout: true).trim()
                    return branch.contains('development') &&
                        (currentBuild.result == null || currentBuild.result == 'SUCCESS')
                }
            }
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'carlemil112-github',
                    usernameVariable: 'GIT_USER', //pragma: allowlist secret
                    passwordVariable: 'GIT_TOKEN' //pragma: allowlist secret
                )]) {
                    sh '''
                        git config user.email "chejsl23@student.aau.dk"
                        git config user.name "carlemil112"
                        git remote set-url origin https://$GIT_USER:$GIT_TOKEN@github.com/carlemil112/mlops_project.git

                        git checkout -f main || true
                        git merge origin/development --no-ff -m "Auto-merge from Jenkins build $BUILD_NUMBER"
                        git push origin main
                    '''
                }
            }
        }


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

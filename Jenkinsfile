pipeline {
    agent any

    options {
        timestamps()
        // For avoiding double checkout
        skipDefaultCheckout()
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
                    credentialsID: 'minio_ass',
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
    stage('Cleanup'){
        post {
            always {
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
}

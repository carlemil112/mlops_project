pipeline {
    agent any

    options {
        timestamps()
        // For avoiding double checkout
        skipDefaultCheckout()
    }

    stages {
        stage('Checkout') {
            steps {
                echo "Using branch: ${env.BRANCH_NAME ?: env.GIT_BRANCH ?: 'unknown'}"
                checkout scm
            }
        }


        // opbygning af docker image
        stage('Build Docker Image') {
            steps {
                echo "Building docker image from SHA: ${env.GIT_COMMIT}"

                sh '''
                    docker --version
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
    }

    post {
        always {
            // cleanup
            sh '''
                SHORT_SHA=$(echo "$GIT_COMMIT" | cut -c1-7)
                docker image rm -f mlops_project_tests:$BUILD_NUMBER mlops_project_tests:$SHORT_SHA || true

            '''
        }
    }
}

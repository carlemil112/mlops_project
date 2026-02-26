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
                echo "Pulling repo from ${BRANCH_NAME}..."
                checkout scm
            }
        }


        // opbygning af docker image
        stage('Build Docker Image') {
            steps {
                sh 'docker --version'
                sh "docker build -t mlops_project_tests:${BUILD_NUMBER} ."
            }
        }

        stage('Run Unit Tests (pytest)') {
            steps {
                sh "docker run --rm mlops_project_tests:${BUILD_NUMBER} python -m pytest -q"
            }
        }
    }

    post {
        always {
            // cleanup
            sh "docker image rm -f mlops_project_tests:${BUILD_NUMBER} || true"
        }
    }
}

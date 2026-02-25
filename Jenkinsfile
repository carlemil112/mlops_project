pipeline {
    agent any

    options {
        timestamps()
        // For avoiding double checkout
        skipDefaultCheckout()
    }

    stages {
        echo "Running on branch ${BRANCH_NAME}..."
        stage('Load Data')
	        steps {
                echo 'Loading Data from remote....'
	    }
        stage('Checkout') {
            stepss {
                checkout scm
            }
        }
        // opbygning af docker image
        stage('Build Docker Image') {
            steps {
		echo 'Building docker image...'
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

version="1.0.0"
repository="192.168.1.2:5001"
imagename="raceboxoverlay"
appname="raceboxoverlay"
tag="latest"
argocd_server="argocd.tommyeriksen.dk"

image="${repository}/${imagename}:${tag}"
altversion="${version}.${env.BUILD_NUMBER}"



podTemplate(label: 'demo-customer-pod', cloud: 'kubernetes', serviceAccount: 'jenkins',
  containers: [
    containerTemplate(name: 'buildkit', image: '192.168.1.2/localbuildkit:latest', ttyEnabled: true, privileged: true),
  ],
  volumes: [
    secretVolume(mountPath: '/etc/.ssh', secretName: 'ssh-home')
  ]) {
node {
        def app

        stage('Clone repository') {
            /* Let's make sure we have the repository cloned to our workspace */

            checkout scm
        }

        stage('Build Docker Image') {
            container('buildkit') {

                sh """
                    /usr/bin/buildctl build \
                    --export-cache type=registry,ref=${repository}/${imagename}:buildcache,mode=max \
                    --import-cache type=registry,ref=${repository}/${imagename}:buildcache \
                    --frontend dockerfile.v0 \
                    --local context=. \
                    --opt platform=linux/amd64,linux/arm64 \
                    --local dockerfile=. \
                    --output type=image,name=${repository}/${imagename}:${tag},push=true,registry.insecure=true,registry.http=true
                """
               milestone(1)
            }

        }
    }
}

properties([[
    $class: 'BuildDiscarderProperty',
    strategy: [
        $class: 'LogRotator',
        artifactDaysToKeepStr: '', artifactNumToKeepStr: '', daysToKeepStr: '', numToKeepStr: '10']
    ]
]);
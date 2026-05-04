FROM eclipse-temurin:17-jre-alpine

ARG BROUTER_VERSION=1.7.9
ENV BROUTER_DIR=/opt/brouter

WORKDIR ${BROUTER_DIR}

# Pull the published BRouter release. The zip extracts to brouter-${VER}/
# containing brouter-${VER}-all.jar, lib/, and profiles2/ (default profiles).
RUN apk add --no-cache --virtual .build-deps curl unzip \
 && curl -fsSL "https://github.com/abrensch/brouter/releases/download/v${BROUTER_VERSION}/brouter-${BROUTER_VERSION}.zip" -o /tmp/brouter.zip \
 && unzip -q /tmp/brouter.zip -d /tmp/brouter \
 && mv /tmp/brouter/brouter-${BROUTER_VERSION}/* . \
 && mv brouter-${BROUTER_VERSION}-all.jar brouter.jar \
 && rm -f BRouterApp*.apk \
 && rm -rf /tmp/brouter /tmp/brouter.zip \
 && apk del .build-deps \
 && mkdir -p customprofiles segments4

EXPOSE 17777

# RouteServer args: <segments-dir> <profiles-dir> <customprofiles-dir> <port> <max-threads>
# Classpath includes lib/* in case the -all jar is not actually a fat jar in this release.
CMD ["java", "-Xmx512M", "-Xms64M", "-DmaxRunningTime=300", \
     "-cp", "brouter.jar:lib/*", "btools.server.RouteServer", \
     "segments4", "profiles2", "customprofiles", "17777", "4"]

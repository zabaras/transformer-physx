var CACHE = 'cache-and-update';

var urlsToCache = [
  
    
      '/403.html',
    
  
    
       
    
  
    
      '/category/',
    
  
    
      '/assets/css/global.css',
    
  
    
      '/',
    
  
    
      '/search.json',
    
  
    
      '/sw.js',
    
  
    
      '/tags/',
    
  
    
      '/tags.json',
    
  
    
      '/category/updates.html',
    
  
    
      '/assets/css/style.css',
    
  
    
      '/sitemap.xml',
    
  
    
      '/robots.txt',
    
  
    
      '/feed.xml',
    
  

  
    '/updates/2021/05/31/transformer-physx-website.html',
  

  
    '/LICENSE',
  
    '/assets/css/customCss.css',
  
    '/assets/images/asbestos_3.jpg',
  
    '/assets/images/koopman.png',
  
    '/assets/images/posts/post_1.jpg',
  
    '/assets/images/results/Cylinder0.webm',
  
    '/assets/images/results/Cylinder1.webm',
  
    '/assets/images/results/Grayscott0.webm',
  
    '/assets/images/results/Grayscott1.webm',
  
    '/assets/images/results/Lorenz0.webm',
  
    '/assets/images/results/cylinder0.mp4',
  
    '/assets/images/results/cylinder1.mp4',
  
    '/assets/images/results/grayscott0.mp4',
  
    '/assets/images/results/grayscott1.mp4',
  
    '/assets/images/results/lorenz0.mp4',
  
    '/assets/images/touch/apple-touch-icon.png',
  
    '/assets/images/touch/chrome-touch-icon-192x192.png',
  
    '/assets/images/touch/icon-128x128.png',
  
    '/assets/images/touch/ms-touch-icon-144x144-precomposed.png',
  
    '/assets/images/transformer.png',
  
    '/assets/images/updates.jpg',
  
    '/assets/js/History.js',
  
    '/assets/js/customJS.js',
  
    '/manifest.json',
  
];

self.addEventListener('install', function(evt) {
  evt.waitUntil(caches.open(CACHE).then(function(cache) {
    cache.addAll(urlsToCache);
  }));
});

self.addEventListener('fetch', function(evt) {
  evt.respondWith(fromCache(evt.request));
  evt.waitUntil(update(evt.request));
});

function fromCache(request) {
  return caches.open(CACHE).then(function(cache) {
    return cache.match(request).then(function(response) {
      if (response != undefined) {
        return response;
      } else {
        return fetchFromInternet(request);
      }
    });
  }).catch(function() {
    return caches.match('/offline.html');
  });
}

function update(request) {
  return caches.open(CACHE).then(function(cache) {
    return fetchFromInternet(request);
  });
}

function fetchFromInternet(request) {
  var fetchRequset = request.clone();
  return fetch(fetchRequset).then(function(response) {
    if (!response || response.status !== 200 || response.type !== 'basic') {
            return response;
    }
    var responseToCache = response.clone();
    caches.open(CACHE).then(function(cache) {
      cache.put(request, responseToCache);
    });
    return response;
  });
}
